image1 = imread('pexels-iriser-1379636.jpg');
image2 = imread('pexels-gochrisgoxyz-1643409.jpg');
image3 = imread('pexels-pixabay-235767.jpg');

[rows, cols, ~] = size(image1);
image2 = imresize(image2, [rows, cols]);
image3 = imresize(image3, [rows, cols]);

midCol = floor(cols / 2);

part1 = image1(:, 1:midCol, :);
part3 = image2(:, 1:midCol, :);
part5 = image3(:, 1:midCol, :);

output_part1 = zeros(size(part1), 'uint8');
output_part1(:, :, 1) = part1(:, :, 1);

output_part3 = zeros(size(part3), 'uint8');
output_part3(:, :, 2) = part3(:, :, 2);

output_part5 = zeros(size(part5), 'uint8');
output_part5(:, :, 3) = part5(:, :, 3);

output_image = [output_part1, output_part3, output_part5];
output_image_text = insertText(output_image, [midCol/4 rows/3], '1', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);
output_image_text = insertText(output_image_text, [midCol + midCol/4 rows/3], '3', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);
output_image_text = insertText(output_image_text, [2*midCol + midCol/4 rows/3], '5', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);

image1_with_line = insertShape(image1, 'Line', [midCol 1 midCol rows], 'Color', 'red', 'LineWidth', 100);
image1_with_text = insertText(image1_with_line, [midCol/4 rows/3], '1', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);
image1_with_text = insertText(image1_with_text, [midCol + midCol/4 rows/3], '2', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);

image2_with_line = insertShape(image2, 'Line', [midCol 1 midCol rows], 'Color', 'red', 'LineWidth', 100);
image2_with_text = insertText(image2_with_line, [midCol/4 rows/3], '3', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);
image2_with_text = insertText(image2_with_text, [midCol + midCol/4 rows/3], '4', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);

image3_with_line = insertShape(image3, 'Line', [midCol 1 midCol rows], 'Color', 'red', 'LineWidth', 100);
image3_with_text = insertText(image3_with_line, [midCol/4 rows/3], '5', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);
image3_with_text = insertText(image3_with_text, [midCol + midCol/4 rows/3], '6', 'FontSize', 200, 'TextColor', 'black', 'BoxColor', 'white', 'BoxOpacity', 1);

combined_images = [image1_with_text, image2_with_text, image3_with_text];

figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
subplot(2, 1, 1), imshow(combined_images);
subplot(2, 1, 2), imshow(output_image_text);

imwrite(output_image_text, '20200204049.png');