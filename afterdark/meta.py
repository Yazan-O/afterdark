import base64
import json
import mimetypes
import os
from pathlib import Path
import random
import re
import time

import httpx

from afterdark import spend


class MetaAPIError(RuntimeError):
    pass


def _key(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise MetaAPIError(f"Missing API key: set {name}")
    return value


def _image_url(image_path):
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0]
    if not mime or not mime.startswith("image/"):
        raise ValueError("Image path must have a recognized image extension")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _sam_stream(response):
    chunks = []
    data = []
    for line in response.iter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            event = json.loads("\n".join(data))
            data.clear()
            kind = event.get("type")
            if kind == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise MetaAPIError("SAM returned an invalid text delta")
                chunks.append(delta)
            elif kind == "response.completed":
                return "".join(chunks)
            elif kind in {"error", "response.failed", "response.incomplete"}:
                raise MetaAPIError("SAM segmentation did not complete")
    raise MetaAPIError("SAM stream ended before response.completed")


REQUEST_TIMEOUT = 120.0


def _request(url, key, payload):
    paid = "api.meta.ai" in url
    if paid:
        spend.check()
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            for attempt in range(4):
                with client.stream(
                    "POST", url,
                    headers={"Authorization": f"Bearer {key}"}, json=payload,
                ) as response:
                    retry = response.status_code == 429 or 500 <= response.status_code < 600
                    wait = 0.5 * 2**attempt + random.uniform(0, 0.25)
                    try:
                        wait = max(wait, min(float(response.headers.get("retry-after", 0)), 65.0))
                    except ValueError:
                        pass
                    if retry and attempt < 3:
                        pass
                    elif not 200 <= response.status_code < 300:
                        raise MetaAPIError(f"Model API request failed: HTTP {response.status_code}")
                    elif payload.get("stream"):
                        text = _sam_stream(response)
                        if paid:
                            spend.record(payload["model"], 1)
                        return text
                    else:
                        response.read()
                        body = response.json()
                        if paid:
                            spend.record(payload["model"], 1, body.get("usage"))
                        return body
                time.sleep(wait)
    except httpx.RequestError:
        raise MetaAPIError("Model API transport failed") from None
    except (ValueError, TypeError, AttributeError):
        raise MetaAPIError("Model API returned an invalid response") from None


def _sam_records(text):
    if not text.strip():
        return []
    body = text.strip()
    if not body.startswith("<0f>"):
        raise MetaAPIError("SAM returned an invalid image frame")
    body = body[4:]
    records = []
    pattern = re.compile(r"(\d+)<\|box;([^|]+)\|><\|mask;(.+?)\|>")
    while body:
        match = pattern.match(body)
        if not match:
            raise MetaAPIError("SAM returned malformed segmentation geometry")
        try:
            box = dict(field.split("=", 1) for field in match[2].split(";"))
            header, encoded = match[3].split(";data=", 1)
            mask = dict(field.split("=", 1) for field in header.split(";"))
            height, width, encoded = encoded.split(",", 2)
            score = float(box["c"])
            coords = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
            frame_w, frame_h = int(box["w"]), int(box["h"])
            if not (
                0 <= score <= 1 and float(mask["c"]) == score
                and mask["x"] == mask["y"] == "0"
                and 0 <= coords[0] <= coords[2] < frame_w
                and 0 <= coords[1] <= coords[3] < frame_h
                and int(height) > 0 and int(width) > 0
                and len(encoded) > 1 and encoded[0] in "~!"
            ):
                raise ValueError
            records.append({
                "object_id": int(match[1]), "box": coords,
                "width": frame_w, "height": frame_h, "score": score,
                "mask": {
                    "height": int(height), "width": int(width),
                    "encoding": "lossless" if encoded[0] == "~" else "one_bit",
                    "data": encoded[1:],
                },
            })
        except (KeyError, ValueError, IndexError):
            raise MetaAPIError("SAM returned invalid box, mask, or confidence fields") from None
        body = body[match.end():]
        if body:
            if not body.startswith(",") or len(body) == 1:
                raise MetaAPIError("SAM returned malformed object separators")
            body = body[1:]
    return records


def sam_segment(image_path, phrase):
    key = _key("MODEL_API_KEY")
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("phrase must be one nonempty noun phrase")
    payload = {
        "model": "sam-3.1",
        "input": [{
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": _image_url(image_path)},
            ],
        }],
        "stream": True,
        "metadata": {"include_confidence": "true"},
    }
    return _sam_records(_request("https://api.meta.ai/v1/responses", key, payload))


def _message(response):
    try:
        message = response["choices"][0]["message"]
        if not isinstance(message, dict):
            raise TypeError
        return message
    except (KeyError, IndexError, TypeError):
        raise MetaAPIError("Model API returned no assistant message") from None


def spark_ask(image_paths, question, model="muse-spark-1.3", max_completion_tokens=None, reasoning_effort=None):
    key = _key("MODEL_API_KEY")
    paths = list(image_paths)
    if not paths or not isinstance(question, str) or not question.strip():
        raise ValueError("Provide images and a nonempty question")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            *({"type": "image_url", "image_url": {"url": _image_url(p)}} for p in paths),
        ]}],
    }
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    response = _request("https://api.meta.ai/v1/chat/completions", key, payload)
    text = _message(response).get("content")
    if not isinstance(text, str) or not text.strip():
        raise MetaAPIError("Muse Spark returned no answer text")
    return text


def spark_open_chat(messages, tools=None):
    key = _key("BASETEN_API_KEY")
    payload = {"model": "meta-models/muse-spark-open-1026", "messages": messages}
    if tools is not None:
        payload["tools"] = tools
    response = _request("https://inference.baseten.co/v1/chat/completions", key, payload)
    _message(response)
    return response
