#!/usr/bin/env bash

badwords_S=("AAAAAA" "AAAAAAlab" "CCCCCC" "DDD")

goodwords_S=("AAAAAA" "BBBBBBBBB" "CCCCCC" "DDD")

for ss in "${!badwords_S[@]}"; do

	bword="${badwords_S[ss]}"
	gword="${goodwords_S[ss]}"

    printf "%s %s\n" "$bword" "$gword"

	find . -type f \( \
		-name "*.py" -o \
		-name "*.sh" -o \
		-name "*.md" -o \
		-name "*.slurm" -o \
		-name "*.json" -o \
		-name "*.yaml" -o \
		-name "*.yml" -o \
		-name "*.txt" -o \
		-name "*.cfg" -o \
		-name "*.ini" \
	\) -print0 | xargs -0 sed -i '' "s/${bword}/${gword}/g"

done