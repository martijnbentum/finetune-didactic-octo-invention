#!/bin/bash
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <target_directory> <transcription> [-s (sampa) | -o (orthographic)]"
  return 1
fi

# goal_dir="orthographic_dutch_960_100000/checkpoint-8100/" 
goal_dir=$1
transcription=$2

cp helper_files/*.json $goal_dir

if [[ $transcription == "-s" ]]; then
  mv $goal_dir"sampa_vocab.json" $goal_dir"vocab.json"
  mv $goal_dir"orthographic_vocab.json" $goal_dir"vocab.json"
  return 0
elif [[ $transcription == "-o" ]]; then
  mv $goal_dir"orthographic_vocab.json" $goal_dir"vocab.json"
  mv $goal_dir"sampa_vocab.json" $goal_dir"vocab.json"
  return 0
else
    echo "transcription should be either -s (sampa) or -o (orthographic)"
    echo "transcription: $transcription"
    echo "Usage: $0 <target_directory> <transcription> [-s (sampa) | -o (orthographic)]"
    return 1
fi
