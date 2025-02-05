% Digital Image Processing Lab Guide
% This guide covers key image processing operations with detailed explanations

%% 1. Basic Plotting and Matrix Operations
x = 1:50;
y = x.^3;
plot(x,y,'.-b'); % Create cubic function plot in blue with points and lines
hold on;         % Keep current plot active for multiple plots

y = (x+10).^3;   
plot(x,y,'.-r'); % Add second cubic function in red
hold off;        
grid on;         % Add grid lines for better visualization

% Matrix Operations Demo
A = floor(rand(5,5)*10);  % Create 5x5 random matrix with values 0-9
B = ones(5,5)*9;         % Create 5x5 matrix filled with 9s
C = A + B;               % Matrix addition
[row, col] = size(C);    % Get matrix dimensions
I = eye(row, col);       % Create identity matrix
D = C.*I;                % Element-wise multiplication (zeros off diagonal)

%% 2. Intensity Level Slicing
% This technique modifies specific ranges of pixel intensities
% Useful for enhancing certain features in an image

img = imread('cameraman.png');
[row, col] = size(img);
output_img = uint8(zeros(row, col));

% Modify pixels in range 101-150 to white (255)
% This helps highlight specific features in the image
for i = 1:row
    for j = 1:col
        if img(i, j) >= 101 && img(i, j) <= 150
            output_img(i, j) = 255;
        else
            output_img(i, j) = img(i, j);
        end
    end
end

%% 3. Histogram Generation
% Histogram shows frequency distribution of pixel intensities
% Important for analyzing image characteristics and performing enhancements

freq = uint8(zeros(1, 256));  % Create array for 256 possible intensity values

for i = 1:row
    for j = 1:col
        temp = img(i, j) + 1;  % +1 for 1-based indexing
        freq(1, temp) = freq(1, temp) + 1;
    end
end

%% 4. Basic Image Transformations
% These transformations modify pixel intensities to enhance image features

% Negative Transformation: S = L-1-R
% Useful for enhancing white or grey details in dark regions
neg_img = uint8(zeros(row, col));
L = 256;  % Number of intensity levels
for i = 1:row
    for j = 1:col
        neg_img(i, j) = L - 1 - img(i, j);
    end
end

%% 5. Advanced Transformations
% Convert to double precision for mathematical operations
img = im2double(img);

% Logarithmic Transformation: S = c * log(1 + R)
% Expands dark pixels, compresses bright pixels
logarithmic_img = zeros(row, col);
c = 1;      % Scaling constant
gamma = 0.05;  % Power law parameter

for i = 1:row
    for j = 1:col
        % Log transformation
        logarithmic_img(i, j) = c * log(1 + img(i, j));
        
        % Power Law (Gamma) Transformation: S = c * R^γ
        % γ < 1: Expand dark regions
        % γ > 1: Expand bright regions
        pow_law_img(i, j) = c * (img(i, j) ^ gamma);
    end
end

%% 6. Thresholding
% Converts grayscale image to binary based on threshold
% Useful for segmentation and object detection

output_img = uint8(zeros(row, col));
threshold = 128;

for i = 1:row
    for j = 1:col
        if img(i, j) >= threshold
            output_img(i, j) = 255;  % White
        else
            output_img(i, j) = 0;    % Black
        end
    end
end

%% 7. Image Resizing and Zooming
% Different methods for changing image size

% Nearest Neighbor Interpolation (Simple Zooming)
zoomed_row = round(row * 2);  % 2x zoom
zoomed_col = round(col * 2);
zoomed_img = uint8(zeros(zoomed_row, zoomed_col));

for i = 1:zoomed_row
    for j = 1:zoomed_col
        % Find nearest neighbor in original image
        nn_i = max(min(round(i/2), row), 1);
        nn_j = max(min(round(j/2), col), 1);
        zoomed_img(i, j) = img(nn_i, nn_j);
    end
end

% Built-in MATLAB resize functions
resized_img = imresize(img, 1.25);        % Scale by factor
resized_img2 = imresize(img, [100 150]);  % Specify dimensions

%% 8. Bit Operations
% Useful for image compression and manipulation
test = uint8([1 2 3]);
right_shift = bitsra(test, 1);  % Right shift by 1 bit
left_shift = bitsll(test, 1);   % Left shift by 1 bit
and_op = bitand(test, 1);       % Bitwise AND
or_op = bitor(test, 4);         % Bitwise OR
xor_op = bitxor(test, 2);       % Bitwise XOR



// 1. Image Intensity Transformations:
// - Negative transformation (inverse)
// - Log transformation (enhances dark pixels)
// - Power-law (gamma) transformation (controls contrast)
// - Linear transformations

// 2. Histogram Processing:
// - Understanding histogram generation
// - What histograms tell us about image characteristics
// - How to use histograms for image enhancement

// 3. Spatial Operations:
// - Intensity level slicing
// - Thresholding
// - Bit-plane operations

// 4. Image Resizing:
// - Nearest neighbor interpolation
// - Understanding scaling factors
// - Different interpolation methods

// 5. Important MATLAB Functions:
// - imread/imwrite
// - imshow/imtool
// - size
// - zeros/ones
// - uint8/im2double conversions
