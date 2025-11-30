# Choom

A project for training and testing Chinese character recognition using CNN models and GNT/MPF dataset parsing.

## Features

* Convert `.gnt` files to PNG visualizations
* Clean, readable CNN architecture for classification
* Data loaders for custom dataset ingestion
* Utilities for experimenting with different model architectures

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Choom supports the handwriting datasets stored in `.gnt` format. The project includes dedicated utilities to parse these files and export them to images.


## Usage

```bash
python infer.py --image your_image.png
```

## License

MIT License
