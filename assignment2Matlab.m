I = imread('pexels-iriser-1379636.jpg');

[row, col, dim] = size(I);

if dim == 3
    I = rgb2gray(I);
end

A = min(I(:));
B = max(I(:));
D = B - A;
M = 2^8 - 1;
R = ((double(I) - double(A)) / double(D)) * double(M) + double(A);
R = uint8(R);

input_hist = zeros(1, 256);
output_hist = zeros(1, 256);

for i = 1:row
    for j=1:col
        temp = I(i,j)+1;
        input_hist(1,temp) = input_hist(1,temp) + 1;
        temp = R(i,j)+1;
        output_hist(1,temp) = output_hist(1,temp) + 1;
    end
end

figure;
subplot(2,2,1),bar(input_hist),title('Histogram of Input Image');
subplot(2, 2, 2), bar(output_hist), title('Histogram of Output Image');
subplot(2, 2, 3), imshow(I), title('Input Image');
subplot(2, 2, 4), imshow(R), title('Output Image');
