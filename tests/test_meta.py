import base64
import json
import uuid

import httpx
import pytest
from PIL import Image

from afterdark import meta


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    values = {name: uuid.uuid4().hex for name in ("MODEL_API_KEY", "BASETEN_API_KEY")}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(meta.time, "sleep", lambda seconds: None)
    return values


@pytest.fixture
def images(tmp_path):
    paths = []
    for suffix, color in (("png", "red"), ("jpg", "blue")):
        path = tmp_path / f"image.{suffix}"
        Image.new("RGB", (3, 2), color).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def transport(monkeypatch):
    original = httpx.Client

    def install(handler):
        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original(*args, **kwargs)

        monkeypatch.setattr(meta.httpx, "Client", client)

    return install


def event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def sam_stream(*deltas, complete=True):
    body = "".join(
        event("response.output_text.delta", {"type": "response.output_text.delta", "delta": delta})
        for delta in deltas
    )
    if complete:
        body += event("response.completed", {"type": "response.completed"})
    return body


def token(object_id=0, payload="VPaz", marker="~"):
    return (
        f"<0f>{object_id}<|box;x1=1;y1=2;x2=3;y2=5;w=16;h=12;c=0.75|>"
        f"<|mask;x=0;y=0;c=0.75;data=4,3,{marker}{payload}|>"
    )


def assert_image_url(url, path, mime):
    prefix, encoded = url.split(",", 1)
    assert prefix == f"data:{mime};base64"
    assert base64.b64decode(encoded, validate=True) == path.read_bytes()


def invoke(api, images):
    if api == "sam":
        return meta.sam_segment(images[0], "person")
    if api == "spark":
        return meta.spark_ask(images, "What is visible?")
    return meta.spark_open_chat([{"role": "user", "content": "Hello"}])


def success_response(api):
    if api == "sam":
        return httpx.Response(200, text=sam_stream(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"choices": [{"message": {"content": "Visible"}}]})


def test_sam_request_shape_and_encoded_mask(transport, credentials, images):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=sam_stream(token()), headers={"content-type": "text/event-stream"})

    transport(handler)
    result = meta.sam_segment(images[0], "person wearing red")
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.meta.ai/v1/responses"
    assert request.headers["authorization"] == f"Bearer {credentials['MODEL_API_KEY']}"
    body = json.loads(request.content)
    content = body["input"][0]["content"]
    assert body == {
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": content}],
        "stream": True,
        "metadata": {"include_confidence": "true"},
    }
    assert len(content) == 2
    assert content[0] == {"type": "input_text", "text": "person wearing red"}
    assert content[1]["type"] == "input_image"
    assert_image_url(content[1]["image_url"], images[0], "image/png")
    assert result == [{
        "object_id": 0,
        "box": [1, 2, 3, 5],
        "width": 16,
        "height": 12,
        "mask": {"width": 3, "height": 4, "encoding": "lossless", "data": "VPaz"},
        "score": 0.75,
    }]


@pytest.mark.parametrize("fragment_size", [1, 7, 29])
def test_sam_fragmented_text_preserves_sparse_ids(transport, images, fragment_size):
    text = token(2) + "," + token(19)[4:]
    body = sam_stream(*(text[index:index + fragment_size] for index in range(0, len(text), fragment_size)))
    transport(lambda request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    result = meta.sam_segment(images[0], "person")
    assert [item["object_id"] for item in result] == [2, 19]
    assert all(item["mask"]["data"] == "VPaz" for item in result)


@pytest.mark.parametrize("marker,encoding", [("~", "lossless"), ("!", "one_bit")])
def test_sam_retains_encoded_mask(transport, images, marker, encoding):
    transport(lambda request: httpx.Response(200, text=sam_stream(token(marker=marker))))
    mask = meta.sam_segment(images[0], "person")[0]["mask"]
    assert mask == {"width": 3, "height": 4, "encoding": encoding, "data": "VPaz"}


def test_sam_empty_completed_response(transport, images):
    transport(lambda request: success_response("sam"))
    assert meta.sam_segment(images[0], "person") == []


@pytest.mark.parametrize("body", [
    sam_stream(token(), complete=False),
    sam_stream("<0f>0<|box;x1=1"),
    sam_stream(token().replace("x1=1", "x1=invalid")),
    sam_stream(token().replace(";c=0.75|>", "|>", 1)),
    'event: response.output_text.delta\ndata: {invalid}\n\n',
    event("error", {"type": "error", "error": {"message": "stream failed"}}),
    event("response.failed", {"type": "response.failed", "response": {"error": {"message": "failed"}}}),
])
def test_sam_rejects_invalid_or_unfinished_stream(transport, images, body):
    transport(lambda request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    with pytest.raises(meta.MetaAPIError):
        meta.sam_segment(images[0], "person")


@pytest.mark.parametrize("model", [None, "caller-selected-model"])
def test_spark_request_and_text_response(transport, credentials, images, model):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Two images."}}]})

    transport(handler)
    kwargs = {} if model is None else {"model": model}
    assert meta.spark_ask(images, "Compare these", **kwargs) == "Two images."
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.meta.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {credentials['MODEL_API_KEY']}"
    body = json.loads(request.content)
    content = body["messages"][0]["content"]
    assert body == {
        "model": "muse-spark-1.3" if model is None else model,
        "messages": [{"role": "user", "content": content}],
    }
    assert content[0] == {"type": "text", "text": "Compare these"}
    assert len(content) == 3
    for item, path, mime in zip(content[1:], images, ("image/png", "image/jpeg")):
        assert item["type"] == "image_url"
        assert_image_url(item["image_url"]["url"], path, mime)


@pytest.mark.parametrize("body", [
    {},
    {"choices": []},
    {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": ""}}]},
    {"choices": [{"message": {"content": "   "}}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": ["text"]}}]},
])
def test_spark_rejects_invalid_text_response(transport, images, body):
    transport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(meta.MetaAPIError):
        meta.spark_ask(images, "Compare these")


@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]])
def test_open_chat_preserves_tool_calls_and_request(transport, credentials, tools):
    messages = [{"role": "system", "content": "Inspect carefully"}, {"role": "user", "content": "Hello"}]
    response = {"id": "response", "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]}}]}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    transport(handler)
    assert meta.spark_open_chat(messages, tools=tools) == response
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://inference.baseten.co/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {credentials['BASETEN_API_KEY']}"
    expected = {"model": "meta-models/muse-spark-open-1026", "messages": messages}
    if tools is not None:
        expected["tools"] = tools
    assert json.loads(request.content) == expected


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": None}]}, {"choices": [{"message": "text"}]}])
def test_open_chat_rejects_missing_message(transport, body):
    transport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(meta.MetaAPIError):
        meta.spark_open_chat([{"role": "user", "content": "Hello"}])


@pytest.mark.parametrize("api", ["sam", "spark", "open"])
def test_retries_eventually_succeed(transport, monkeypatch, images, api):
    attempts = []
    delays = []
    monkeypatch.setattr(meta.time, "sleep", delays.append)

    def handler(request):
        attempts.append(request)
        if len(attempts) <= 3:
            return httpx.Response([429, 500, 599][len(attempts) - 1], json={"error": {"message": "retry"}})
        return success_response(api)

    transport(handler)
    invoke(api, images)
    assert len(attempts) == 4
    assert len(delays) == 3
    assert all(delay > 0 for delay in delays)
    assert delays[0] < delays[1] < delays[2]


@pytest.mark.parametrize("api", ["sam", "spark", "open"])
@pytest.mark.parametrize("status", [429, 500, 599])
def test_retry_exhaustion_has_four_total_attempts(transport, images, api, status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(status, json={"error": {"message": "retry"}})

    transport(handler)
    with pytest.raises(meta.MetaAPIError):
        invoke(api, images)
    assert len(attempts) == 4


@pytest.mark.parametrize("api", ["sam", "spark", "open"])
@pytest.mark.parametrize("status", [400, 401, 404])
def test_permanent_http_errors_do_not_retry(transport, images, api, status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(status, json={"error": {"message": "rejected"}})

    transport(handler)
    with pytest.raises(meta.MetaAPIError):
        invoke(api, images)
    assert len(attempts) == 1


@pytest.mark.parametrize("api,variable", [("sam", "MODEL_API_KEY"), ("spark", "MODEL_API_KEY"), ("open", "BASETEN_API_KEY")])
def test_missing_credentials_are_clear_runtime_errors(monkeypatch, transport, images, api, variable):
    monkeypatch.delenv(variable)

    def handler(request):
        pytest.fail("Missing credentials must fail before sending a request")

    transport(handler)
    with pytest.raises(meta.MetaAPIError, match=variable) as error:
        invoke(api, images)
    assert isinstance(error.value, RuntimeError)


@pytest.mark.parametrize("api", ["spark", "open"])
def test_invalid_json_is_an_api_error(transport, images, api):
    transport(lambda request: httpx.Response(200, text="{invalid"))
    with pytest.raises(meta.MetaAPIError):
        invoke(api, images)
