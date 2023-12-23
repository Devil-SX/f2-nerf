# Resize images.
DATASET_PATH=$1

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_2

pushd "$DATASET_PATH"/images_2
ls | xargs -P 8 -I {} mogrify -resize 50% {}
popd

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4

pushd "$DATASET_PATH"/images_4
ls | xargs -P 8 -I {} mogrify -resize 25% {}
popd

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_8

pushd "$DATASET_PATH"/images_8
ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
popd