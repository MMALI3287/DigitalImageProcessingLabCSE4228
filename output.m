% Load three images
image1 = imread('vibrant-garden-filled-with-color.png');
image2 = imread('view-mountain-with-dreamy-aesthe.png');
image3 = imread('garden-with-flowers-tree-with-su.png');

% Resize images to the same size for consistency
[rows, cols, ~] = size(image1);
image2 = imresize(image2, [rows, cols]);
image3 = imresize(image3, [rows, cols]);

% Split each image vertically into two equal parts
midCol = floor(cols / 2);

part1 = image1(:, 1:midCol, :); % First half of image1
part3 = image2(:, 1:midCol, :); % First half of image2
part5 = image3(:, 1:midCol, :); % First half of image3

% Create output parts with specific color channels
output_part1 = zeros(size(part1), 'uint8'); % Initialize as black
output_part1(:, :, 1) = part1(:, :, 1);    % Red channel only

output_part3 = zeros(size(part3), 'uint8'); % Initialize as black
output_part3(:, :, 2) = part3(:, :, 2);    % Green channel only

output_part5 = zeros(size(part5), 'uint8'); % Initialize as black
output_part5(:, :, 3) = part5(:, :, 3);    % Blue channel only

% Combine the parts horizontally to form the output image
output_image = [output_part1, output_part3, output_part5];

% Display the images
figure;
subplot(2, 2, 1), imshow(image1), title('Image 1');
subplot(2, 2, 2), imshow(image2), title('Image 2');
subplot(2, 2, 3), imshow(image3), title('Image 3');
subplot(2, 2, 4), imshow(output_image), title('Output Image');

% Save the output image
imwrite(output_image, 'output_image.jpg');