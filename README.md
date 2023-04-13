# Install
```
./install.sh
```
# Testing

1. Generate feature for champions
```
python generate_features.py
```

2. Run prediction
```
python test.py <image_directory> <output_text_path>
```

Example:
```
python test.py data/test_data/test_images output.txt
```
