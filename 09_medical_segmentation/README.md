# TP9: Segmentation of Follicles

This practical work focuses on image segmentation for the analysis of ovarian follicles in histological images. The main objective is to extract and quantify different parts of a follicle, including:

1. **Antrum**: The central cavity of the follicle (appears white in the original image)
2. **Theca**: The ring region around the antrum where vascularization occurs
3. **Vascularization**: The blood vessels within the theca
4. **Granulosa cells**: Located between the antrum and vascularization

## Approach

The segmentation approach combines thresholding and mathematical morphology techniques:

1. **Antrum Segmentation**: Extract the central white region using thresholding on the blue component
2. **Theca Segmentation**: Extract the ring region around the antrum using binary dilation
3. **Vascularization Segmentation**: Extract regions with low blue component within the theca
4. **Granulosa Cell Segmentation**: Extract the region between the antrum and vascularization

## Quantification

After segmentation, the code calculates various measurements:
- Area of each component in pixels
- Proportion of vascularization and granulosa cells relative to the total follicle area

## Files

- `follicle_segmentation.py`: Main implementation of the segmentation and quantification algorithms
- `main.py`: Runner script to execute the follicle segmentation process

## Usage

You can run the code using either of these commands:

```bash
# Run directly from the TP9 directory
python main.py

# Or use the project runner script
python run.py --all-tp 9

# Or specify the specific script
python run.py TP9_Follicle_Segmentation/main.py
```

## Results

The segmentation results are displayed as:
1. Original follicle image
2. Segmented antrum
3. Segmented theca
4. Segmented vascularization
5. Segmented granulosa cells
6. Combined color-coded segmentation (antrum in blue, granulosa in green, vascularization in red)

Results are saved in the `plots/TP9_Follicle_Segmentation` directory. 