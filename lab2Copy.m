O = imread('pexels-dreamypixel-547119.jpg'); 
K = imfinfo('pexels-dreamypixel-547119.jpg');

% Convert the image to grayscale if it's a color image (24-bit depth)
if K.BitDepth == 24
    I = rgb2gray(O);
end

% Get the size of the image
[r, c] = size(I);

% Convert the image to double once for efficient processing
I_double = double(I);

% Initialize images for 2, 4, and 8 thresholds
I2 = uint8(I_double > 128) * 255;  % 2-thresholded (direct vectorization)
I3 = uint8(ceil(I_double / 64) * 64);  % 4-thresholded (direct vectorization)
I4 = uint8(ceil(I_double / 32) * 32);  % 8-thresholded (direct vectorization)

% Negative Image Transformation (vectorized)
L = 256;  % The image has intensity ranging from 0-255, so intensity level is 256
neg_img = uint8(L - 1 - I);  % Direct negation

% Logarithmic Transformation
c = 20;  % Increased constant to enhance the result
logarithmic_img = uint8(c * log(1 + I_double));  % Vectorized log transformation

% Inverse Logarithmic Transformation
inv_logarithmic_img = uint8(exp(I_double / c) - 1);  % Vectorized inverse log transformation

% Power Law Transformation
gamma = 0.4;  % Increased gamma to avoid too dark images
pow_law_img = uint8(c * (I_double .^ gamma));  % Vectorized power law transformation

% Display the images
figure;
subplot(3, 3, 1), imshow(I), title('Greyscale Image');
subplot(3, 3, 2), imshow(I2), title('2-Thresholded Image');
subplot(3, 3, 3), imshow(I3), title('4-Thresholded Image');
subplot(3, 3, 4), imshow(I4), title('8-Thresholded Image');
subplot(3, 3, 5), imshow(O), title('Original Image');
subplot(3, 3, 6), imshow(neg_img), title('Negative Image');
subplot(3, 3, 7), imshow(logarithmic_img), title('Logarithmic Image');
subplot(3, 3, 8), imshow(inv_logarithmic_img), title('Inverse Logarithmic Image');
subplot(3, 3, 9), imshow(pow_law_img), title('Power Law Image');
