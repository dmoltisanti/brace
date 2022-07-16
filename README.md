# Introduction

This repository contains the dataset published with the ECCV 2022 paper "**BRACE: The Breakdancing Competition Dataset for Dance Motion Synthesis**".

## What is BRACE?

- A new dataset for audio-conditioned dance motion synthesis
- Focuses on breakdancing sequences
- Contains high quality annotations for complex body poses and dance movements

### BRACE at a glance

| Property                   | Value        |
| -------------------------- | ------------ |
| Frames                     | 334,538      |
| Manually annotated frames  | 26,676 (8%)  |
| Duration                   | 3h 32m       |
| Dancers                    | 64           |
| Videos                     | 81           |
| Sequences                  | 465          |
| Segments                   | 1,352        |
| Avg. segments per sequence | 2.91         |
| Avg. sequence duration     | 27.48s       |
| Avg. segment duration      | 9.45s        |

# Download

## Keypoints

You can download our keypoints [here](https://github.com/dmoltisanti/brace/releases/download/v1.0/dataset.zip).

Notice that keypoints are split into **segments**, i.e. the shorter clips that compose 
a sequence (please refer to our paper for more details). 
We provide code to load these keypoints both as segments or as sequences (see below).

Keypoints are JSON files organised in folders as follows:

```bash
├── year
│ ├── video_id
│ │ ├── video_id_start-end_movement.json
```

Where `video_id_start-end_movement` denote the corresponding information about the segment. 
For example `3rIk56dcBTM_1234-1330_powermove.json` indicates:
- video_id: `3rIk56dcBTM`
- start: `1234`
- end: `1330`
- movement: `powermove`

Start/end are in frames. Movement can be one of `(toprock, footwork, powermove)`. 

## Videos and frames

We used `youtube-dl` to download the videos from YouTube (links are provided in [video_info.csv](https://github.com/dmoltisanti/brace/blob/main/videos_info.csv))
using 

```bash
format: bestvideo[ext=mp4],bestaudio[ext=m4a]
``` 

To extract frames we simply used `ffmpeg` without re-encoding the videos:

```bash
ffmpeg -i ./videos/{} ./frames/{}/img-%06d.png'.format(video_id)
```

## Manually annotated keypoints

https://github.com/dmoltisanti/brace/releases/download/mk_v1.0/manual_keypoints.zip

# Pytorch dataset

# Citation

```bibtex
@InProceedings{moltisanti22brace,
author = "Moltisanti, Davide and Wu, Jinyi and Bo, Dai and Loy, Chen Change",
title = "BRACE: The Breakdancing Competition Dataset for Dance Motion Synthesis",
booktitle = "European Conference on Computer Vision (ECCV)",
year = "2022"
}
```

# Supplementary material video

[![BRACE - supplementary material](https://img.youtube.com/vi/IIL7yeALxaQ/0.jpg)](http://www.youtube.com/watch?v=IIL7yeALxaQ)
