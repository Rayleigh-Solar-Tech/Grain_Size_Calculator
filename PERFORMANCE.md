# Grain Size Calculator - CPU Performance Optimization

## High-Performance CPU Configuration

The application is now optimized for maximum performance on strong CPUs:

### SAM Model Configuration
- **Model**: `sam_l.pt` (Large model for best accuracy)
- **Resolution**: 1536px max side (high resolution for detailed analysis)
- **Processing**: CPU-optimized with all available cores

### CPU Optimizations Applied

1. **Multi-threading**: Uses all available CPU cores
2. **Memory Optimization**: Disabled gradients for inference
3. **BLAS Optimization**: Enabled MKL-DNN for faster CPU operations (Windows compatible)
4. **Inference Mode**: Optimized PyTorch settings for CPU inference

### Performance Features

- **Real-time Progress**: Shows processing time for each variant
- **Memory Efficient**: Automatic image downscaling when needed
- **Multi-variant Processing**: Parallel enhancement parameter testing
- **Batch Optimization**: Efficient processing pipeline

### Expected Performance

**Processing Time per Variant** (approximate):
- Small images (< 1MP): 30-60 seconds
- Medium images (1-4MP): 1-3 minutes  
- Large images (4-10MP): 3-8 minutes

**Total Analysis Time** for 6 variants:
- Small images: 3-6 minutes
- Medium images: 6-18 minutes
- Large images: 18-48 minutes

### Hardware Recommendations

**Minimum Requirements**:
- 4-core CPU (Intel i5/AMD Ryzen 5)
- 8GB RAM
- 2GB free disk space

**Recommended for Best Performance**:
- 8+ core CPU (Intel i7/i9 or AMD Ryzen 7/9)
- 16GB+ RAM
- SSD storage
- Multiple CPU cores will significantly improve processing speed

### Usage Tips for Best Performance

1. **Close other applications** during analysis
2. **Use SSD storage** for input/output files
3. **Ensure adequate cooling** for sustained performance
4. **Process images in batches** using CLI mode for efficiency

### Memory Usage

The application automatically manages memory by:
- Downscaling large images to 1536px
- Processing variants sequentially (not in parallel)
- Clearing intermediate results between variants
- Using efficient data structures

This configuration provides the best possible accuracy while maintaining reasonable processing times on strong CPUs.