# MSCOCO-30K evaluation dataset

MSCOCO-30K contains 30,000 images selected from the COCO 2014 `train` and
`val` splits. We use the filename list released with
[PerCo-SD](https://github.com/Nikolai10/PerCo):

- list page: <https://github.com/Nikolai10/PerCo/blob/master/res/doc/coco_names.txt>
- raw list: <https://raw.githubusercontent.com/Nikolai10/PerCo/master/res/doc/coco_names.txt>

The list contains exactly 30,000 unique JPEG names. Its SHA-256 checksum is:

```text
3106be3aea6800ca7fd5a6c01e067bee91fe7d31130b093c09db99c870c58d77
```

The selected subset has 9,038 images from `val2014` and 20,962 from
`train2014`. Images are stored under their bare twelve-digit COCO IDs, for
example `000000000009.jpg`, rather than the original split-prefixed name.

## 1. Download COCO 2014 and the filename list

Choose an arbitrary working directory; no fixed local path is required.

```bash
set -euo pipefail

DATA_ROOT=/path/to/MSCOCO_30K
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

curl -fL http://images.cocodataset.org/zips/val2014.zip -o val2014.zip
curl -fL http://images.cocodataset.org/zips/train2014.zip -o train2014.zip
unzip -q val2014.zip
unzip -q train2014.zip

curl -fL \
  https://raw.githubusercontent.com/Nikolai10/PerCo/master/res/doc/coco_names.txt \
  -o coco_names.txt
printf '%s  %s\n' \
  3106be3aea6800ca7fd5a6c01e067bee91fe7d31130b093c09db99c870c58d77 \
  coco_names.txt | sha256sum --check --strict
```

The extracted COCO directories should contain 40,504 validation images and
82,783 training images.

## 2. Select the 30,000 images

COCO names files as `COCO_<split>2014_<id>.jpg`, whereas the PerCo list only
contains `<id>.jpg`. The following deterministic selection restores the split
prefix for lookup and removes it in the copied subset:

```bash
set -euo pipefail
cd /path/to/MSCOCO_30K
mkdir -p selected

val_count=0
train_count=0
missing_count=0
while IFS= read -r name; do
  if [[ -f "val2014/COCO_val2014_${name}" ]]; then
    cp "val2014/COCO_val2014_${name}" "selected/${name}"
    val_count=$((val_count + 1))
  elif [[ -f "train2014/COCO_train2014_${name}" ]]; then
    cp "train2014/COCO_train2014_${name}" "selected/${name}"
    train_count=$((train_count + 1))
  else
    printf 'Missing image: %s\n' "$name" >&2
    missing_count=$((missing_count + 1))
  fi
done < coco_names.txt

selected_count=$(find selected -maxdepth 1 -type f -name '*.jpg' | wc -l)
printf 'selected=%s val=%s train=%s missing=%s\n' \
  "$selected_count" "$val_count" "$train_count" "$missing_count"

[[ "$selected_count" -eq 30000 ]]
[[ "$val_count" -eq 9038 ]]
[[ "$train_count" -eq 20962 ]]
[[ "$missing_count" -eq 0 ]]
```

## 3. Produce the 256x256 evaluation images

From the RAE-CoD repository, run:

```bash
python dataset/resize_and_crop.py \
  --input-dir /path/to/MSCOCO_30K/selected \
  --output-dir /path/to/MSCOCO_30K/selected_res256 \
  --size 256 \
  --expected-count 30000
```

For every source image, the script performs the following fixed preprocessing:

1. convert the image to RGB;
2. resize its shortest side to 256 pixels with Pillow Lanczos interpolation,
   preserving the aspect ratio and using floor rounding for the other side;
3. take a `256x256` center crop with integer-centered offsets; and
4. save the result losslessly as PNG, changing `<id>.jpg` to `<id>.png`.

There is no random resizing, cropping, or flipping. The resulting directory
contains exactly 30,000 RGB PNGs, each with spatial size `256x256`.

The raw COCO directories and ZIP archives may be deleted after both `selected/`
and `selected_res256/` have been verified. COCO images remain subject to the
COCO dataset terms; this repository only provides the preprocessing procedure.
