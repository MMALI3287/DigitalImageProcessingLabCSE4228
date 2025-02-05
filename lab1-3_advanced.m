%% Comprehensive Image Processing Guide
% This guide covers spatial domain enhancement, filters, and transformations

%% 1. Image Enhancement in Spatial Domain
% Spatial domain refers to direct manipulation of image pixels
% Basic formula: g(x,y) = T[f(x,y)]
% where f(x,y) is input image, T is operator, g(x,y) is output image

%% 2. Gray Level Transformations

% 2.1 Linear Transformation
img = imread('cameraman.tif');
img = im2double(img);  % Convert to double for mathematical operations

% Identity Transformation (S = r)
identity_img = img;  % No change to pixel values

% Image Negation (S = L-1-r)
negative_img = 1 - img;  % For normalized image, L-1 = 1

% Display results
figure;
subplot(131); imshow(img); title('Original');
subplot(132); imshow(identity_img); title('Identity');
subplot(133); imshow(negative_img); title('Negative');

%% 3. Logarithmic Transformation
% S = c * log(1 + r)
% Expands dark pixels while compressing bright pixels

c = 1;  % Scaling constant
log_img = c * log(1 + img);

% Normalize to [0,1] range
log_img = log_img / max(log_img(:));

figure;
subplot(121); imshow(img); title('Original');
subplot(122); imshow(log_img); title('Log Transform');

%% 4. Power Law (Gamma) Transformation
% S = c * r^gamma
% gamma < 1: Expand dark regions
% gamma > 1: Expand bright regions

gamma1 = 0.5;  % Expand dark regions
gamma2 = 2.0;  % Expand bright regions
c = 1;

power_img1 = c * img.^gamma1;
power_img2 = c * img.^gamma2;

figure;
subplot(131); imshow(img); title('Original');
subplot(132); imshow(power_img1); title('\gamma = 0.5');
subplot(133); imshow(power_img2); title('\gamma = 2.0');

%% 5. Piecewise Linear Transformation

% 5.1 Thresholding
threshold = 0.5;  % For normalized image
threshold_img = img > threshold;

% 5.2 Contrast Stretching
% Maps input range [r1,r2] to output range [s1,s2]
r1 = 0.2; r2 = 0.8;  % Input range
s1 = 0.0; s2 = 1.0;  % Output range

contrast_img = img;
% Below r1
mask = img <= r1;
contrast_img(mask) = (s1/r1) * img(mask);
% Between r1 and r2
mask = img > r1 & img <= r2;
contrast_img(mask) = ((s2-s1)/(r2-r1)) * (img(mask)-r1) + s1;
% Above r2
mask = img > r2;
contrast_img(mask) = ((1-s2)/(1-r2)) * (img(mask)-r2) + s2;

%% 6. Bit Plane Slicing
% Separate 8-bit image into individual bit planes
img_uint8 = imread('cameraman.tif');
bit_planes = zeros(size(img_uint8, 1), size(img_uint8, 2), 8);

for bit = 1:8
    bit_planes(:,:,bit) = bitget(img_uint8, bit);
end

%% 7. Spatial Filtering

% 7.1 Low Pass Filter (Smoothing)
% Reduces noise and blur edges
kernel_size = 3;
lpf = ones(kernel_size) / (kernel_size^2);  % Average filter
img_lpf = conv2(img, lpf, 'same');

% 7.2 High Pass Filter (Sharpening)
% Enhances edges
hpf = [-1 -1 -1;
       -1  8 -1;
       -1 -1 -1];  % Laplacian filter
img_hpf = conv2(img, hpf, 'same');

% 7.3 Median Filter
% Good for salt-and-pepper noise
img_median = medfilt2(img, [3 3]);

%% 8. Noise Filtering
% Add noise for demonstration
noisy_img = imnoise(img, 'gaussian', 0, 0.01);  % Gaussian noise
sp_img = imnoise(img, 'salt & pepper', 0.05);   % Salt & pepper noise

% Apply different filters
gaussian_filtered = imgaussfilt(noisy_img, 1);  % Gaussian filter
median_filtered = medfilt2(sp_img, [3 3]);      % Median filter

%% 9. Template Matching using Convolution
% Create simple template (e.g., vertical edge)
template = [-1 1;
           -1 1;
           -1 1];

% Perform template matching
matched = conv2(img, template, 'same');
matched_norm = mat2gray(abs(matched));  % Normalize for visualization

%% 10. Custom Convolution Implementation
function output = custom_conv2d(image, kernel)
    [h, w] = size(image);
    [kh, kw] = size(kernel);
    
    % Padding sizes
    pad_h = floor(kh/2);
    pad_w = floor(kw/2);
    
    % Pad image
    padded = padarray(image, [pad_h pad_w], 'replicate');
    
    % Output image
    output = zeros(h, w);
    
    % Perform convolution
    for i = 1:h
        for j = 1:w
            % Extract region of interest
            roi = padded(i:i+kh-1, j:j+kw-1);
            % Apply kernel
            output(i,j) = sum(roi(:) .* kernel(:));
        end
    end
end

% Example usage of custom convolution:
kernel = fspecial('gaussian', [3 3], 0.5);
filtered_img = custom_conv2d(img, kernel);


// Let me explain the key concepts from each section:

// Spatial Domain Enhancement:


// Works directly with pixel values
// Each pixel is modified based on some transformation function
// Can be local (neighborhood) or global (entire image)


// Linear Transformations:


// Image Negation: Inverts intensities, useful for enhancing white details in dark regions
// Identity: No change, useful as reference
// Simple to implement but limited in capability


// Logarithmic Transformation:


// Expands dark pixel values while compressing bright ones
// Useful for enhancing details in dark regions
// Good for images with large dynamic range


// Power Law (Gamma):


// Controls contrast through gamma parameter
// γ < 1: Enhance dark regions
// γ > 1: Enhance bright regions
// Widely used in display devices


// Filters:


// Low Pass: Smooths image, reduces noise
// High Pass: Enhances edges, increases sharpness
// Median: Excellent for removing salt-and-pepper noise
// Each has specific use cases and trade-offs

// Would you like me to elaborate on any of these topics or provide additional examples? I can also show you how different parameters affect the results of these transformations.
// Also, for your exam, it's important to understand:

// When to use each type of transformation
// The mathematical basis behind each operation
// How to implement these operations efficiently
// Common applications and real-world uses