# ENF matching

The example code from my blog post about ENF matching: ["How to date a recording using background electrical noise"](https://robertheaton.com/enf). This script predicts when a target recording was taken by comparing its background electrical noise to a reference recording.

## Requirements

* Python3
* Virtualenv (or similar)
* Curl (for downloading sample files) (can also download manually)

## Usage

### Setup

```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Download sample files from https://github.com/ghuawhu/ENF-WHU-Dataset:

```
./bin/download-examnple-files
```

### Run

```
source venv/bin/activate
python3 main.py
```

Should output:

```
<snip>
True value is 71458
Best prediction is 71460
```
