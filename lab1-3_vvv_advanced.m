%% Complete Digital Image Processing Guide
% Comprehensive coverage of all topics with implementations

%% 1. FUNDAMENTAL CONCEPTS
% Image representation
% A digital image is a 2D matrix of pixels
% For grayscale: Each pixel is a single intensity value
% For RGB: Each pixel has three color channel values

%% 2. IMAGE TYPES AND CONVERSIONS
% Reading and converting between different image types
img_rgb = imread('peppers.png');
img_gray = rgb2gray(img_rgb);         % Convert RGB to grayscale
img_double = im2double(img_gray);      % Convert to double [0,1]
img_uint8 = im2uint8(img_double);      % Convert to uint8 [0,255]
img_binary = imbinarize(img_gray);     % Convert to binary

%% 3. INTENSITY TRANSFORMATIONS

% 3.1 Basic Linear Transformations
function output = basic_linear_transform(input_img, slope, intercept)
    output = slope * input_img + intercept;
    output = min(max(output, 0), 1);  % Clip to [0,1]
end

% 3.2 Piecewise Linear Transform
function output = piecewise_linear(input_img, points)
    % points is nx2 matrix of (x,y) coordinates defining the transform
    output = zeros(size(input_img));
    
    for i = 1:size(points,1)-1
        mask = input_img >= points(i,1) & input_img <= points(i+1,1);
        slope = (points(i+1,2) - points(i,2)) / (points(i+1,1) - points(i,1));
        output(mask) = points(i,2) + slope * (input_img(mask) - points(i,1));
    end
end

% 3.3 Negative Transform
neg_img = 1 - img_double;

% 3.4 Log Transform with Normalization
function output = log_transform(input_img, c)
    if nargin < 2
        c = 1;
    end
    output = c * log(1 + input_img);
    output = output / max(output(:));  % Normalize
end

% 3.5 Power Law (Gamma) Transform
function output = power_law_transform(input_img, gamma, c)
    if nargin < 3
        c = 1;
    end
    output = c * input_img.^gamma;
    output = output / max(output(:));  % Normalize
end

%% 4. HISTOGRAM PROCESSING

% 4.1 Histogram Calculation
function hist = calculate_histogram(img, bins)
    if nargin < 2
        bins = 256;
    end
    hist = zeros(1, bins);
    for i = 1:numel(img)
        bin = floor(img(i) * (bins-1)) + 1;
        hist(bin) = hist(bin) + 1;
    end
end

% 4.2 Histogram Equalization
function output = histogram_equalization(input_img)
    % Calculate histogram
    hist = imhist(input_img);
    
    % Calculate cumulative histogram
    cum_hist = cumsum(hist);
    
    % Normalize
    cum_hist_norm = cum_hist / cum_hist(end);
    
    % Apply transformation
    output = cum_hist_norm(round(input_img * 255) + 1);
end

% 4.3 Histogram Matching (Specification)
function output = histogram_matching(input_img, ref_img)
    % Get cumulative histograms
    hist_input = imhist(input_img);
    hist_ref = imhist(ref_img);
    
    cum_hist_input = cumsum(hist_input) / sum(hist_input);
    cum_hist_ref = cumsum(hist_ref) / sum(hist_ref);
    
    % Create mapping
    map = zeros(256, 1);
    j = 1;
    for i = 1:256
        while j < 256 && cum_hist_ref(j) < cum_hist_input(i)
            j = j + 1;
        end
        map(i) = j - 1;
    end
    
    % Apply mapping
    output = map(round(input_img * 255) + 1) / 255;
end

%% 5. SPATIAL FILTERING

% 5.1 Convolution Implementation
function output = convolution2D(input_img, kernel)
    [h, w] = size(input_img);
    [kh, kw] = size(kernel);
    
    % Padding
    ph = floor(kh/2);
    pw = floor(kw/2);
    padded = padarray(input_img, [ph pw], 'replicate');
    
    output = zeros(h, w);
    for i = 1:h
        for j = 1:w
            region = padded(i:i+kh-1, j:j+kw-1);
            output(i,j) = sum(region(:) .* kernel(:));
        end
    end
end

% 5.2 Common Spatial Filters
% Averaging (Low Pass)
lpf_avg = ones(3,3) / 9;

% Gaussian (Low Pass)
lpf_gaussian = fspecial('gaussian', [3 3], 0.5);

% Laplacian (High Pass)
hpf_laplacian = [0 1 0; 1 -4 1; 0 1 0];

% Sobel (Edge Detection)
sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];

% Prewitt (Edge Detection)
prewitt_x = [-1 0 1; -1 0 1; -1 0 1];
prewitt_y = [-1 -1 -1; 0 0 0; 1 1 1];

%% 6. NOISE AND NOISE REDUCTION

% 6.1 Add Different Types of Noise
noisy_gaussian = imnoise(img_double, 'gaussian', 0, 0.01);
noisy_salt_pepper = imnoise(img_double, 'salt & pepper', 0.05);
noisy_speckle = imnoise(img_double, 'speckle', 0.04);

% 6.2 Noise Reduction Filters
% Median Filter
function output = median_filter(input_img, window_size)
    [h, w] = size(input_img);
    pad = floor(window_size/2);
    padded = padarray(input_img, [pad pad], 'replicate');
    output = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            window = padded(i:i+window_size-1, j:j+window_size-1);
            output(i,j) = median(window(:));
        end
    end
end

% Adaptive Median Filter
function output = adaptive_median_filter(input_img, max_window)
    [h, w] = size(input_img);
    output = input_img;
    
    for i = 1:h
        for j = 1:w
            window_size = 3;
            while window_size <= max_window
                % Get window
                pad = floor(window_size/2);
                if i-pad < 1 || i+pad > h || j-pad < 1 || j+pad > w
                    break;
                end
                window = input_img(max(1,i-pad):min(h,i+pad), ...
                                 max(1,j-pad):min(w,j+pad));
                
                % Stage A
                med = median(window(:));
                min_val = min(window(:));
                max_val = max(window(:));
                
                if med > min_val && med < max_val
                    % Stage B
                    if input_img(i,j) > min_val && input_img(i,j) < max_val
                        output(i,j) = input_img(i,j);
                    else
                        output(i,j) = med;
                    end
                    break;
                else
                    window_size = window_size + 2;
                end
            end
        end
    end
end

%% 7. EDGE DETECTION

% 7.1 Gradient-based Edge Detection
function [magnitude, direction] = edge_gradient(input_img)
    % Sobel operators
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % Compute gradients
    Gx = convolution2D(input_img, sobel_x);
    Gy = convolution2D(input_img, sobel_y);
    
    % Compute magnitude and direction
    magnitude = sqrt(Gx.^2 + Gy.^2);
    direction = atan2(Gy, Gx);
end

% 7.2 Canny Edge Detection Implementation
function edges = canny_edge_detection(input_img, low_thresh, high_thresh)
    % 1. Gaussian smoothing
    gaussian = fspecial('gaussian', [5 5], 1.4);
    smoothed = convolution2D(input_img, gaussian);
    
    % 2. Compute gradients
    [magnitude, direction] = edge_gradient(smoothed);
    
    % 3. Non-maximum suppression
    suppressed = non_max_suppression(magnitude, direction);
    
    % 4. Double thresholding
    strong_edges = suppressed >= high_thresh;
    weak_edges = suppressed >= low_thresh & suppressed < high_thresh;
    
    % 5. Edge tracking by hysteresis
    edges = hysteresis_tracking(strong_edges, weak_edges);
end

%% 8. MORPHOLOGICAL OPERATIONS

% 8.1 Basic Morphological Operations
function output = dilate(input_img, se)
    [h, w] = size(input_img);
    [sh, sw] = size(se);
    ph = floor(sh/2);
    pw = floor(sw/2);
    padded = padarray(input_img, [ph pw], 0);
    output = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            region = padded(i:i+sh-1, j:j+sw-1);
            output(i,j) = max(region(:) .* se(:));
        end
    end
end

function output = erode(input_img, se)
    [h, w] = size(input_img);
    [sh, sw] = size(se);
    ph = floor(sh/2);
    pw = floor(sw/2);
    padded = padarray(input_img, [ph pw], 1);
    output = ones(h, w);
    
    for i = 1:h
        for j = 1:w
            region = padded(i:i+sh-1, j:j+sw-1);
            output(i,j) = min(region(:) + (1 - se(:)));
        end
    end
end

% 8.2 Advanced Morphological Operations
function output = opening(input_img, se)
    output = dilate(erode(input_img, se), se);
end

function output = closing(input_img, se)
    output = erode(dilate(input_img, se), se);
end

%% 9. FREQUENCY DOMAIN PROCESSING

% 9.1 Fourier Transform
function [magnitude, phase] = fourier_analysis(input_img)
    F = fft2(input_img);
    F_shifted = fftshift(F);
    magnitude = abs(F_shifted);
    phase = angle(F_shifted);
end

% 9.2 Frequency Domain Filtering
function output = freq_domain_filter(input_img, filter)
    % Transform to frequency domain
    F = fft2(input_img);
    F_shifted = fftshift(F);
    
    % Apply filter
    filtered = F_shifted .* filter;
    
    % Transform back to spatial domain
    output = real(ifft2(ifftshift(filtered)));
end

%% 10. TEMPLATE MATCHING

function [response, locations] = template_matching(input_img, template, threshold)
    % Normalize cross correlation
    response = normxcorr2(template, input_img);
    
    % Find peaks above threshold
    [y, x] = find(response > threshold);
    locations = [x y];
    
    % Adjust for template size
    [th, tw] = size(template);
    locations = locations - [floor(tw/2) floor(th/2)];
end

%% UTILITY FUNCTIONS

% Image Quality Metrics
function psnr_val = calculate_psnr(original, processed)
    mse = mean((original(:) - processed(:)).^2);
    psnr_val = 10 * log10(1 / mse);  % For normalized images
end

function ssim_val = calculate_ssim(original, processed)
    % Simplified SSIM implementation
    % Constants
    K1 = 0.01;
    K2 = 0.03;
    L = 1;  % Dynamic range
    
    C1 = (K1*L)^2;
    C2 = (K2*L)^2;
    
    % Means
    mu1 = mean2(original);
    mu2 = mean2(processed);
    
    % Variances and covariance
    sigma1_sq = var(original(:));
    sigma2_sq = var(processed(:));
    sigma12 = mean2((original - mu1).*(processed - mu2));
    
    ssim_val = ((2*mu1*mu2 + C1)*(2*sigma12 + C2))/...
               ((mu1^2 + mu2^2 + C1)*(sigma1_sq + sigma2_sq + C2));
end