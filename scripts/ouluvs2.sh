#!/bin/bash

urls=(
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/readme"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_mouth_mp4_phrase.zip"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_mouth_mp4_digit.zip"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/cropped_audio_dat.zip"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/landmark-sentence.zip"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/transcript_digit_phrase"
    "http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/transcript_sentence.zip"
)

# dir=data/datasets/ouluvs2
# mkdir -p $dir
# for url in "${urls[@]}"; do
#     wget --user $USER --password $PASSWORD -P $dir $url
# done

mkdir -p data/datasets/ouluvs2/orig
for i in {1..53}; do
    wget --user $USER --password $PASSWORD http://www.ee.oulu.fi/research/imag/OuluVS2/OuluVS2-zip/orig_s$i.zip
    unzip -d data/datasets/ouluvs2 orig_s$i.zip
    rm orig_s$i.zip
done
