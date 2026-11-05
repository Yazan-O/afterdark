# Afterdark

**Meta's models read a brain-inspired camera at night, with no retraining.**

A car drives through Zurich at night carrying two kinds of camera, each as a stereo pair: ordinary cameras and event cameras. Afterdark uses the left one of each. The event camera is a neuromorphic sensor that sees motion the way a retina does. Meta's **SAM 3.1** finds people on both. Meta's **Muse Image** turns the event camera's flickers into pictures, and frame after frame, into a video. Meta's **Muse Spark** and **Muse Glimmer** check what was painted against what the sensors recorded.

[Try Afterdark]([LIVE_URL]) | [Watch the film]([VIDEO_URL])

![SAM 3.1 finds no one in the dark camera frame. In the event camera's view it outlines the person.](docs/img/hero_camera_blind.png)

*A person at a tram stop beside the road, Zurich at night. Left: the camera frame as recorded; SAM 3.1 finds no one. Middle: the same frame brightened for your eyes; SAM 3.1 still finds no one. Right: 10 ms of the event camera at the same instant; SAM 3.1, given the same one-word prompt "person", outlines the person. The dashed box is where the dataset's own label puts them.*

## The problem: night

At night an ordinary camera can lose most of its picture to darkness, and long exposures blur anything that moves. A person stepping toward the road is exactly the kind of thing that gets lost. Cars and robots that must work after dark need a sensor that keeps seeing.

## The camera that keeps seeing

An event camera has no global shutter. Every pixel works on its own, like a cell in the retina. When the logarithm of the brightness it sees changes by more than a set threshold, it fires a spike-like signal, an "event": pixel position, time to the microsecond, and whether it got brighter or darker. Pixels that see no change mostly stay silent, apart from sensor noise.

That gives it three gifts at night ([Gallego et al., 2020](https://arxiv.org/abs/1904.08405)). Motion blur drops sharply, because there is no frame exposure to wait for, though pixels respond more slowly in very dark scenes ([Hu et al., 2021](https://arxiv.org/abs/2006.07722)). It copes with dark streets and bright headlights in the same scene: the survey cites a dynamic range of about 140 dB, against about 60 dB for a standard camera. And it reports change, which on a moving car comes mostly from motion and from lights switching or flickering.

The price: its output is a cloud of flickers, not a picture. People find it hard to read, and the vision models most teams use were trained on pictures, not events.

## The usual way is a whole research pipeline

Event-vision research has built special tools for this, often chained together:

1. **Simulators** turn ordinary videos into fake events, because real labelled event data is scarce: ESIM ([Rebecq et al., 2018](https://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf)) and v2e ([Hu et al., 2021](https://arxiv.org/abs/2006.07722)).
2. **Reconstruction networks**, trained on that simulated data, turn events back into grey images: E2VID ([Rebecq et al., 2019](https://arxiv.org/abs/1906.07165)).
3. **Detectors trained on events** find cars and people, usually trained per dataset and per task.

Each is its own model, often its own training run and its own expertise. Direct event-based methods also exist, and they too are trained for the event data they run on.

## What we found

We asked Meta's general models directly, with no event-specific training.

**SAM 3.1 sees people on both cameras.** We give SAM 3.1 the ordinary camera frame (remapped into the event camera's view) and an image built from 10 to 50 ms of events, with the same prompt, "person". It outlines people on both, with no retraining. On some dark frames, the event view is where it finds the person at all (top figure).

![The same moment on both cameras, with SAM 3.1's outlines on each.](docs/img/two_sensors.png)

**Muse Image paints the night from events.** We give Muse Image a 10 ms event frame and a colour reference image that is not a picture of this scene; no camera frame of the scene goes in the request. It paints a readable night street, and in the examples below the parked cars, lane lines, lamps and houses sit where the events put them. It is a generated picture: colours, textures and anything the events did not show are invented.

![Event frame, Muse Image painting and the real camera, side by side.](docs/img/events_to_picture.png)

**And then a video, made from events.** Muse Image paints ten pictures for every second of driving. Played in order, as keyframes with soft crossfades, they become a night video generated from an event camera's flickers.

![A short loop of Muse Image paintings, keyframes with crossfades, beside their event frames.](docs/img/events_to_video.gif)

![The classic route beside ours: the camera, an E2VID reconstruction and a Muse Image painting.](docs/img/classic_vs_muse.png)

*E2VID reads 1.5 s of signed, timestamped events up to each moment, through a network trained on simulated data. Muse Image reads one 10 ms event frame and a colour reference from another street. They take different inputs, so this is a side-by-side look, not a contest.*

**The agent keeps the painting honest.** A painting can leave things out. Afterdark's agent, Muse Spark through the Meta Model API or Muse Glimmer running on our own NVIDIA H200, compares the painting with both sensors. SAM 3.1 outlines objects on the camera and on the painting, plain arithmetic flags camera objects the painting lacks, and the agent asks Muse Image to repaint that patch, guided by the camera's box. In the recorded climax, Muse Spark found a tram the painting had left out and had it repainted.

![How Afterdark works: two sensors, SAM 3.1 on both, Muse Image painting from events, frames to video, and the agent checking the result.](docs/img/pipeline.svg)

## Who it is for

- **Teams building night perception** for cars, delivery robots and drones: in our recorded examples, one segmentation model worked on a normal camera and a neuromorphic one, which points to a second sensor without a second AI stack.
- **Researchers with event data and no event-trained models:** SAM 3.1 outlines and Muse Image pictures, straight from events, without building simulators and reconstruction networks first.
- **Anyone reviewing night drives:** see the event camera's view as a picture a person can read, and see what the painting invented.

## Where this goes

Event cameras already ship in industrial and automotive sensors, and their output looks like the spikes of a brain-inspired, spiking neural network. Today each of them needs its own trained models. If general models like SAM 3.1 and Muse Image can read them directly, a car or robot can add a neuromorphic camera for night and fast motion without building a second AI stack. The same two-sensor approach fits any setting where light is poor and motion matters: night roads, warehouses, drones at dusk.

## The data

[DSEC](https://dsec.ifi.uzh.ch/) ([Gehrig et al., 2021](https://arxiv.org/abs/2103.06011)): a car driving through Zurich and around Switzerland, day and night, with a stereo pair of Prophesee Gen3.1 event cameras (640 x 480) and a stereo pair of FLIR Blackfly S colour cameras (1440 x 1080, 20 frames per second), about 4.5 cm apart across the two kinds ([paper](https://arxiv.org/pdf/2103.06011), p. 3). Every camera frame and event in Afterdark is recorded data; the paintings are generated, and playback samples the events. People in its labelled detection set ([Gehrig and Scaramuzza, Nature 2024](https://www.nature.com/articles/s41586-024-07409-w)) guided where we looked.

## The research behind it

Afterdark grew out of a PhD on event-based vision. During the hackathon we also ran a research track on finding pedestrians from events alone, on the Prophesee Gen1 driving benchmark ([de Tournemire et al., 2020](https://arxiv.org/abs/2001.08499)). We scored the strongest released event-camera detectors with one evaluation script on the official test split; an independent re-run reproduced every score within 0.01 AP. The best, Ev-DTAD-B, reached 39.6 pedestrian AP. Those detectors were trained for that dataset. Afterdark asks how far Meta's general models get with no event training at all.

## What we measured

All numbers below come from the run folders of this build, on recorded DSEC night drives.

| Question | Result |
| --- | --- |
| Labelled night pedestrians that SAM 3.1 outlined on both the camera and the event view (box overlap 0.5 or more) | 21 of 96 |
| Night people SAM 3.1 found on the brightened camera, and on 10 ms event frames, in a 37-person probe | 30 and 18 of 37; the top figure is one found only on events |
| Cars Muse Image placed where the camera saw them, one 6 s street moment | 274 of 352, against 135 for paintings of a different frame |
| People the painting left out, before and after camera-guided repair by the agent | 216 and 12, of 334 camera sightings |
| Time for one SAM 3.1 call from a laptop | about 1.2 s, so the clip runs frame by frame, not in real time |

## Each Meta model and its job

- **SAM 3.1** (`sam-3.1`) outlines people, cars and trams on the camera frame and on event images. See [meta.py](afterdark/meta.py), [people.py](pipeline/people.py) and [sam_common.py](pipeline/sam_common.py).
- **Muse Image** (`muse-image-1.0`) paints night streets from event frames and repaints patches the agent asks for. See [paint_moment.py](pipeline/paint_moment.py) and [agent.py](afterdark/agent.py).
- **Muse Spark 1.3** (`muse-spark-1.3`) is the agent behind the inspect button. It chooses its own checks. See the [Spark trace](evidence/spark/trace_spark_crosswalk_run2.json).
- **Muse Glimmer 30B** runs as an open-weights agent on our own NVIDIA H200. It follows a step plan computed by code and writes the notes. See [below](#muse-glimmer-on-our-lab-machine).

## Try it locally

From this directory:

```sh
python -m pip install -e .
python -m afterdark serve
```

Open `http://127.0.0.1:8080`. Recorded playback uses local assets and needs no key. Fresh SAM 3.1 and Muse Image calls need `MODEL_API_KEY`, `AFTERDARK_SOURCE` and `AFTERDARK_STYLE`; see [server.py](afterdark/server.py). Keep credentials outside the repository.

## Muse Glimmer on our lab machine

Muse Glimmer 30B ran on our lab machine's NVIDIA H200 through llama.cpp at commit `bee23322f1baa3d2dd16b8339bed6aff4a8eb663`. The [sanitized launch record](evidence/glimmer/setup.txt), [GPU and weight evidence](evidence/glimmer/setup.txt) and [trace](evidence/glimmer/trace_glimmer_crosswalk.json) identify the setup. The [probe](evidence/glimmer/probe.json) records about 30 generated tokens per second.

Across 5 runs, 66 of 66 tool calls parsed cleanly. The raw calls are in [raw_crosswalk_run1.jsonl](evidence/glimmer/raw_crosswalk_run1.jsonl), [raw_crosswalk_run2.jsonl](evidence/glimmer/raw_crosswalk_run2.jsonl), [raw_crosswalk_run3.jsonl](evidence/glimmer/raw_crosswalk_run3.jsonl), [raw_street_run1.jsonl](evidence/glimmer/raw_street_run1.jsonl) and [raw_street_run2.jsonl](evidence/glimmer/raw_street_run2.jsonl). This counts valid tool arguments. It does not score findings.

The recorded trace is `trace_glimmer_crosswalk.json`, shipped at [web/moments/crosswalk/agent.glimmer.json](web/moments/crosswalk/agent.glimmer.json). Glimmer follows the code's step plan. It writes the notes and the report. Keep its report text off screen. It contains a wrong sentence about unmatched cars. The tool results preserve the counts.

The [weight evidence](evidence/glimmer/setup.txt) records these SHA256 hashes for the Q4_K_M files:

```text
Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e
mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6
```

The weights come from [Meta's Muse Glimmer GGUF repository](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF), as recorded in the [sanitized setup](evidence/glimmer/setup.txt). The following direct-server equivalent uses the recorded commit, build settings and server arguments. Run it on a Linux NVIDIA H200 machine with CUDA 12.8, a C++ compiler, Git, CMake, Ninja and curl installed. Other GPU architectures require a different `CMAKE_CUDA_ARCHITECTURES` value. These commands replace the private lab launcher and its shared-machine guards and SSH tunnel.

```sh
mkdir -p "$HOME/afterdark-glimmer"
cd "$HOME/afterdark-glimmer"
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout bee23322f1baa3d2dd16b8339bed6aff4a8eb663
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build llama.cpp/build -j 24 --target llama-server
curl -fL --retry 3 -o Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
curl -fL --retry 3 -o mmproj-Muse-Glimmer-30B-Q4_K_M.gguf https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
printf '%s\n' '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e  Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf' 'f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6  mmproj-Muse-Glimmer-30B-Q4_K_M.gguf' | sha256sum -c -
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$PWD/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
llama.cpp/build/bin/llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf -ngl 99 -c 32768 -np 1 --jinja --temp 1.0 --top-p 0.95 --top-k 64 --host 127.0.0.1 --port 8091 --alias Muse-Glimmer-30B
```

In another shell, from this Afterdark directory after preparing the moment below:

```sh
export GLIMMER_URL=http://127.0.0.1:8091/v1
export GLIMMER_MODEL=Muse-Glimmer-30B
export AFTERDARK_STYLE="$PWD/work/style.png"
python -m afterdark agent work/moments/crosswalk --route glimmer --held 68
```

Glimmer inference is local; the agent's SAM checks and Muse Image repaints still require `MODEL_API_KEY` and paid services. The public sources and archive layouts were checked; a fresh model build and full paid rebuilding are **UNVERIFIED**.

## Built during the hackathon

Mohamad Yazan Sadoun built Afterdark as a solo entry. The work includes event export and calibration mapping, painting prompts and retries, masks and blur, playback preparation, model clients, inspection tools, verdict arithmetic, the spend ledger, the server, the viewer and film tooling. Their source is in [pipeline/](pipeline/), [afterdark/](afterdark/), [web/](web/) and [film/](film/).

The recordings, pretrained models and third-party libraries predate this entry. [pyproject.toml](pyproject.toml) lists package dependencies. [Licences and credits](#licences-and-credits) covers batch dependencies, tools and licence boundaries.

## Running your own drive

The DSEC exporter accepts:

```text
<DSEC_ROOT>/<sequence>/
  ev/events.h5
  ev/rectify_map.h5
  img/*.png
  image_timestamps.txt
```

Install batch dependencies in your processing environment:

```sh
python -m pip install -e .
python -m pip install opencv-python h5py hdf5plugin PyYAML
```

The selection below comes from [moments.sh](pipeline/moments.sh). From this directory in Bash or Git Bash, download the left-camera inputs listed on the [official DSEC download page](https://dsec.ifi.uzh.ch/dsec-datasets/download/). The event archive contains `events.h5` and `rectify_map.h5` at its root; the image archive contains numbered PNGs at its root. The commands extract them into the layout above.

```sh
mkdir -p work/downloads work/dsec/zurich_city_09_c/ev work/dsec/zurich_city_09_c/img work/moments
curl -fL --retry 3 -o work/downloads/events_left.zip https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_09_c/zurich_city_09_c_events_left.zip
curl -fL --retry 3 -o work/downloads/images_rectified_left.zip https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_09_c/zurich_city_09_c_images_rectified_left.zip
curl -fL --retry 3 -o work/dsec/zurich_city_09_c/image_timestamps.txt https://download.ifi.uzh.ch/rpg/DSEC/train/zurich_city_09_c/zurich_city_09_c_image_timestamps.txt
python -m zipfile -e work/downloads/events_left.zip work/dsec/zurich_city_09_c/ev
python -m zipfile -e work/downloads/images_rectified_left.zip work/dsec/zurich_city_09_c/img
python pipeline/export_moment.py --root work/dsec --seq zurich_city_09_c --t0 58305557568 --seconds 9 --out work/moments/crosswalk
```

The historical style file's public origin was not preserved in the run evidence. For a reproducible alternative, use the camera panel from frame 0 of DSEC's [Zurich city 10b preview](https://dsec.ifi.uzh.ch/wp-content/uploads/2021/05/zurich_city_10_b.gif), a different sequence from the crosswalk. The 576x144 preview has three panels; crop the rightmost 192x144 camera panel. This alternative changes the style input and will not recreate the recorded paintings. It carries the DSEC credit and licence below.

```sh
curl -fL --retry 3 -o work/style-source.gif https://dsec.ifi.uzh.ch/wp-content/uploads/2021/05/zurich_city_10_b.gif
python -c "import hashlib; from pathlib import Path; p=Path('work/style-source.gif'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'df69cc781a7a5ce638b015cebfeec49ee79a4a06967a74577c17222a9339760a'"
python -c "from PIL import Image; im=Image.open('work/style-source.gif'); im.seek(0); assert im.size == (576,144); im.convert('RGB').crop((384,0,576,144)).save('work/style.png')"
```

Set `MODEL_API_KEY` in your process environment before the following paid preparation steps. These commands generate new assets and package them into `web/moments/crosswalk`; preserve the recorded assets separately if needed.

```sh
python pipeline/paint_moment.py work/moments/crosswalk --style work/style.png --workers 1 --out paint --step 1
python pipeline/people.py work/moments/crosswalk
python pipeline/blur_faces.py work/moments/crosswalk
python pipeline/stabilise.py work/moments/crosswalk
python pipeline/pack_web.py work/moments/crosswalk
```

The exporter fetches DSEC calibration. Remapping uses rotation and intrinsics without depth-based parallax correction. Review generated content, camera blur and object alignment before packaging. [pack_web.py](pipeline/pack_web.py) packages processed frames into the viewer layout.

## What is real and what is generated

| View | Source and changes |
| --- | --- |
| Events | Recorded light changes with measured positions and times. Playback points are sampled. Activity maps are accumulated and smoothed. |
| Painting | Muse Image output. Colour, texture and areas without event activity are invented. Playback adds drift and blending. |
| Camera | A separate recording, remapped into the event framing, processed for blur and brightened for display. |

Event activity constrains a painting. It does not prove an object's identity. D exposes activity in a smoothed map. A segmentation mask is a model estimate.

## API sources

Every request shape in [meta.py](afterdark/meta.py) follows Meta's documentation, read live during the build.

| Endpoint, model or field | Page read |
| --- | --- |
| Meta base `https://api.meta.ai/v1`, `MODEL_API_KEY`, models `muse-spark-1.3` and `sam-3.1` | https://dev.meta.ai/docs/models |
| `POST https://api.meta.ai/v1/responses`, Bearer authorization, JSON `model` and `input` | https://dev.meta.ai/docs/api-reference/responses/create-response |
| SAM `input[].type=message`, `role=user`, `content[].type=input_text`, `text`, `type=input_image`, string `image_url`; base64 data URLs; `stream=true` | https://dev.meta.ai/docs/sam/segmenting |
| SAM scores requested by `metadata.include_confidence="true"` (a string); default lossless mask encoding | https://dev.meta.ai/docs/sam/segmenting#confidence |
| SSE `response.output_text.delta`, `delta`, terminal `response.completed`; empty completed lane means no matches | https://dev.meta.ai/docs/sam/reading-segmentation |
| SAM frame `<0f>`, integer object id, `box` fields `x1,y1,x2,y2,w,h,c`; `mask` fields `x,y,c,data`; data header height,width; `~` lossless / `!` one_bit; base85 payload | https://dev.meta.ai/docs/sam/reading-segmentation |
| Boxes use inclusive corners; encoded masks can be decoded using the optional vendor parser | https://dev.meta.ai/docs/sam/client-libraries |
| Meta Chat Completions `model`, `messages`, `role`, `content`; result `choices[0].message.content`; HTTP rate-limit backoff | https://dev.meta.ai/docs/protocols/chat-completions |
| `POST https://api.meta.ai/v1/chat/completions`, Bearer authorization and JSON body | https://dev.meta.ai/docs/api-reference/chat-completions/create-chat-completion |
| Vision `content[].type=text`, `text`; `type=image_url`, nested `image_url.url` with MIME/base64 data URL; multiple images | https://dev.meta.ai/docs/image-understanding |
| Baseten `POST https://inference.baseten.co/v1/chat/completions`, `BASETEN_API_KEY`, Bearer authorization, `model`, `messages`, `choices[0].message`, `finish_reason` | https://docs.baseten.co/inference/model-apis/overview |
| Baseten `tools[].type=function`, nested `function.name,description,parameters`; response `message.tool_calls`, `id`, `function.name`, JSON string `function.arguments` | https://docs.baseten.co/inference/function-calling |
| Baseten vision `image_url.url`, base64 media accepted within request-body limits; Muse Spark Open vision support | https://docs.baseten.co/inference/model-apis/vision |
| Requested `meta-models/muse-spark-open-1026` | https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details (supplied signed-in transcription: `hackathon/competition/participant_guide.md`) |

## Licences and credits

[DSEC](https://dsec.ifi.uzh.ch/), Gehrig et al., *DSEC: A Stereo Event Camera Dataset for Driving Scenarios*, supplies the Zurich moments in [web/moments/index.json](web/moments/index.json). DSEC is [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). Derived material in `web/moments` is distributed under that same licence, including event images, sampled event files, camera derivatives, paintings, masks and activity maps derived from the recordings. It is excluded from the code's MIT grant.

Afterdark's changes are interval selection, event accumulation and sampling, calibration remapping, painting generation and release processing such as resizing, blending and face blur. Sequence and timestamp metadata identify the source interval. Dataset authors do not endorse Afterdark.

Original Afterdark code is [MIT](LICENSE), Copyright (c) 2026 Mohamad Yazan Sadoun. The fonts in `web/fonts/` are IBM Plex Mono under the SIL Open Font License 1.1, with its text in [web/fonts/OFL.txt](web/fonts/OFL.txt).

### Dependencies

These are package dependencies, batch dependencies and development tools. They are installed separately; their licences do not become MIT through this project.

- HTTPX: [BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md).
- Pillow: [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE).
- NumPy: [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt), plus notices for bundled binary components.
- OpenCV: [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE). The opencv-python packaging is [MIT](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt); wheels include [third-party notices](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt).
- h5py: [BSD-3-Clause](https://github.com/h5py/h5py/blob/master/LICENSE). Bundled HDF5 retains its own notices.
- hdf5plugin: [MIT](https://github.com/silx-kit/hdf5plugin/blob/main/LICENSE). Bundled HDF5 filters and compression libraries have [separate licences](https://github.com/silx-kit/hdf5plugin/blob/main/doc/information.rst#license).
- PyYAML: [MIT](https://github.com/yaml/pyyaml/blob/main/LICENSE).
- websocket-client, film browser control: [Apache-2.0](https://github.com/websocket-client/websocket-client/blob/master/LICENSE).
- Matplotlib, film audio generation: [Matplotlib licence and included component notices](https://matplotlib.org/stable/project/license.html).
- SciPy, film audio generation: [BSD-3-Clause and bundled component notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt).
- Google Chrome, film rendering tool installed separately: [Chrome terms](https://www.google.com/chrome/terms/) and the component notices supplied with the installed browser.
- FFmpeg, film encoding tool installed separately: [licence depends on the build configuration](https://ffmpeg.org/legal.html). Retain the notices supplied with the selected build.
- meta_sam_parser: [SAM License](https://github.com/meta-models/meta-sam/blob/main/LICENSE), as shipped in the installed parser distribution. It is used to decode segmentation masks.
- setuptools, build tool: [MIT](https://github.com/pypa/setuptools/blob/main/LICENSE).
- pytest, test tool: [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE).
- Python: [PSF licence and included component notices](https://docs.python.org/3/license.html).
- IBM Plex Mono, including the files in `web/fonts/`: [SIL Open Font License 1.1](https://github.com/IBM/plex/blob/master/LICENSE.txt), Copyright IBM Corp., reserved font name "Plex".

HTTPX and the other packages install their own dependencies. Retain the licence files shipped with those distributions when redistributing an environment or binary. This source list is not a wheel-bundle inventory.

### Models

Muse Image, SAM 3.1 and Muse Spark are accessed through Meta's hosted service. Muse Glimmer weights and llama.cpp retain their upstream terms. The Glimmer Q4_K_M files come from `meta-models/Muse-Glimmer-30B-GGUF`, as recorded in [sanitized setup](evidence/glimmer/setup.txt). The [weight evidence](evidence/glimmer/setup.txt) records their SHA256 hashes. Muse Spark Open has an optional client route in [meta.py](afterdark/meta.py). No model weights are included in this source tree.
