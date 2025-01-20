I = imread('pexels-pixabay-235767.jpg');

[row, col, dim] = size(I);

if dim == 3
    I = rgb2gray(I);
end

bit1 = zeros(row, col);
bit2 = zeros(row, col);
bit7 = zeros(row, col);
bit8 = zeros(row, col);

for i = 1:row
    for j = 1:col
        bit1(i, j) = mod(I(i, j), 2); 
        bit2(i, j) = mod(floor(I(i, j) / 2), 2); 
        bit7(i, j) = mod(floor(I(i, j) / 64), 2); 
        bit8(i, j) = mod(floor(I(i, j) / 128), 2); 
    end
end

bit1 = uint8(bit1 * 255);
bit2 = uint8(bit2 * 255);
bit7 = uint8(bit7 * 255);
bit8 = uint8(bit8 * 255);

figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1, 5, 1), imshow(I), title('Input Image');
subplot(1, 5, 2), imshow(bit1), title('1st Bit Plane (LSB)');
subplot(1, 5, 3), imshow(bit2), title('2nd Bit Plane');
subplot(1, 5, 4), imshow(bit7), title('7th Bit Plane');
subplot(1, 5, 5), imshow(bit8), title('8th Bit Plane (MSB)');


imwrite(bit1, 'T2_Output1_20200204049.png');
imwrite(bit2, 'T2_Output2_20200204049.png');
imwrite(bit7, 'T2_Output3_20200204049.png');
imwrite(bit8, 'T2_Output4_20200204049.png');