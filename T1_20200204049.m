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

hist_I = zeros(1, 256);
hist_R = zeros(1, 256);

for i = 1:row
    for j=1:col
        temp = I(i,j)+1;
        hist_I(1,temp) = hist_I(1,temp) + 1;
        temp = R(i,j)+1;
        hist_R(1,temp) = hist_R(1,temp) + 1;
    end
end

figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2, 2, 1), bar(hist_I), title('Input Histogram');
subplot(2, 2, 2), bar(hist_R), title('Output Histogram');
subplot(2, 2, 3), imshow(I), title('Input Image');
subplot(2, 2, 4), imshow(R), title('Output Image');

fig = getframe(gcf);
output = frame2im(fig);

imwrite(output, 'T1_Output1_20200204049.png');
