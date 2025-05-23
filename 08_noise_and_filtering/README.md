# TP8: Image Processing - Random Noise Generation

This practical work focuses on generating, visualizing, and analyzing different types of random noise commonly found in images:

1. **Uniform Noise**
   - Generated using a uniform distribution
   - Values distributed evenly between two bounds

2. **Gaussian Noise**
   - Generated using a normal distribution
   - Bell-shaped distribution around a mean value

3. **Salt and Pepper Noise**
   - Random black and white pixels
   - Models sudden intensity spikes and dead pixels

4. **Exponential Noise**
   - Generated using an exponential distribution
   - Models decay processes in imaging

## Files

- `random_noise.py`: Implementation and visualization of random noise types

## Usage

Run the script to generate noise images and their histograms:

```bash
python TP8_Compression/random_noise.py
```

## Output

Results will be saved in the `plots/TP8_Compression/` directory, including:
- `random_noise_types.png`: Visualization of the four noise types
- `random_noise_histograms.png`: Histograms showing the distribution of each noise type

## Visualization

The script displays:
1. The generated noise images in grayscale
2. The histograms of each noise type showing their probability distributions 