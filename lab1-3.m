x = 1:50;
y = x.^3;
plot(x,y,'.-b'); % plotting the first equation
hold on; % holding the figure to wait.

y = (x+10).^3; % deriving the y for the second equation
plot(x,y,'.-r'); % plotting the first equation
hold off; % do not wait any more

grid on;

bar(x, y)

A = floor(rand(5,5)*10);
B = ones(5,5)*9;
C = A + B;
[row, col] = size(C);
I = eye(row, col);
D = C.*I;
disp(D)

img = imread('cameraman.png');

[row, col] = size(img);

output_img = uint8(zeros(row, col));

% For this program, let's assume we would like to change every contrast level between 101-150
% to 255, and keep the rest identical to input image. 
% you can try experimenting with different range and values.

for i = 1:row
    for j = 1:col
        % condition to check if the intensity level falls within the range
        if img(i, j) >= 101 && img(i, j) <= 150
            output_img(i, j) = 255;
        else
            output_img(i, j) = img(i, j);
        end
    end
end

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(output_img);
title('Intensity Level Slicing Output');


img = imread('cameraman.png');

[row, col] = size(img);

% Histogram is just frequency count of intensity level displayed in a bar chart
% Our image has intensity ranging from 0-255, meaning we have to count frequency of 256 values.
% Experiment with different gray scale image and see the result.

freq = uint8(zeros(1, 256));

for i = 1:row
    for j = 1:col
        temp = img(i, j) + 1; % +1 because intensity level starts at 0, but we will store it in a 1 indexing based matrix
        freq(1, temp) = freq(1, temp) + 1;
    end
end

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
bar(freq);
title('Histogram');

img = imread('cameraman.png');

[row, col] = size(img);

neg_img = uint8(zeros(row, col)); % zeros returns double matrix, we are working with uint8 image
idt_img = uint8(zeros(row, col));

L = 256; % The image has intensity ranging from 0-255, so intensity level is 256

for i = 1:row
    for j = 1:col
        % Negative transformation
        % S = L - 1 - R
        neg_img(i, j) = L - 1 - img(i, j);
        
        % Identity transformation
        % S = R, makes an exact copy of input image
        idt_img(i, j) = img(i, j);
    end
end

figure;
subplot(1, 3, 1);
imshow(img);
title('Original Image');

subplot(1, 3, 2);
imshow(neg_img);
title('Negative');

subplot(1, 3, 3);
imshow(idt_img);
title('Identity');


img = imread('cameraman.png');
img = im2double(img); % convert the uint8 image to double

[row, col] = size(img);

logarithmic_img = zeros(row, col);
inv_logarithmic_img = zeros(row, col);
pow_law_img = zeros(row, col);

c = 1; % try to experiment with different values
gamma = 0.05; % try to experiment with different values

for i = 1:row
    for j = 1:col
        % logarithmic transformation
        % S = C * log (1 + R) 
        logarithmic_img(i, j) = c * log (1 + img(i, j));
        
        % inverse logarithmic transformation
        % as the name implies, totally inverse of logarithmic transformation
        inv_logarithmic_img(i, j) = exp(img(i, j) / c) - 1;
        
        % power law transformation
        % S = C * R ^ gamma
        pow_law_img(i, j) = c * (img(i, j) ^ gamma);
    end
end

figure;
subplot(2, 2, 1);
imshow(img);
title('Original Image');

subplot(2, 2, 2);
imshow(logarithmic_img);
title('Logarithmic Transformation');

subplot(2, 2, 3);
imshow(inv_logarithmic_img);
title('Inverse Logarithmic Transformation');

subplot(2, 2, 4);
imshow(pow_law_img);
title('Power Law Transformation');


img = imread('cameraman.png');

[row, col] = size(img);

output_img = uint8(zeros(row, col));

% Thresholding can be used to change the intensity level of image
% Input image has intensity level from 0-255, total 256 levels.
% For this program, let's assume we want to change the intensity level to only 2 values. So we take only 0 and 255.

% What we can do is, assume a threshold. For example, In this program, we will change every value less than 128, to 0 
% And every value equal or higher than 128, to 255.
% Keep in mind, lowering intensity level will cause loss of information in output image.

% You can experiment with different threshold and Intensity level in output image

for i = 1:row
    for j = 1:col
        % condition to check if the intensity level is above or equals to 128
        if img(i, j) >= 128
            output_img(i, j) = 255;
        else
            output_img(i, j) = 0;
        end
    end
end

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(output_img);
title('Thresholded Image');

% Resize the image
I = imread("circuit.tif");

figure,
subplot(221),imshow(I);

ScaleFactor = 1.25;

J = imresize(I, ScaleFactor);

subplot(222),imshow(J);

K = imresize(I,[100 150]);

subplot(223),imshow(K);

L = imresize(I,ScaleFactor,"nearest");

subplot(224),imshow(L);


% Shrinking the image to 1/2
I = imread('cameraman.tif');
K= imfinfo('cameraman.tif');
if(K.BitDepth ==24)
I=rgb2gray(I);
end
[r,c] = size(I);
I2(1:r/2, 1:c/2) = I(1:2:r, 1:2:c);
figure,
subplot(121),imshow(I);
subplot(122),imshow(I2);

test = uint8([1 2 3]);

% This code will right shift all the elements of test matrix by 1 bit.
% First parameter of this function takes a matrix (test in this example)
% Second parameter of this function takes how many bit do we want to shift right (1 in this example).
a = bitsra(test, 1);

% This code will left shift all the elements of test matrix by 1 bit.
% First parameter of this function takes a matrix (test in this example)
% Second parameter of this function takes how many bit do we want to shift left (1 in this example).
b = bitsll(test, k);

% You can also perform bitwise operation in matlab.
% this will perform bitwise and operation will all the elements of test matrix with 1.
% First parameter of this function takes a matrix (test in this example)
% Second parameter of this function takes with what we want to perfrom bitwise-and with (1 in this example)
c = bitand(test, 1); 

% Similarly we can perform bitwise-or bitwise-xor and so on
d = bitor(test, 4);
e = bitxor(test, 2);

% imtool shows and image like imshow
% But it shows the pixel position and it's value when you hover your mouse in any pixel of the image.
I = imread('cameraman.png');
imtool(I);


img = imread('cameraman.png');

[row, col, dim] = size(img);

% We will 2x zoom the image. So the rows and columns will increase by 2x.
zoomed_row = round(row * 2);
zoomed_col = round(col * 2);

% My input image is uint8 based (ie. intensity levels are 0-255), so type-casting output from zeros into uint8
zoomed_img = uint8(zeros(row, col, dim));

for i = 1:zoomed_row
    for j = 1:zoomed_col

        % i and j are used to iterate the output image. For each pixel (combination of i, j) we need to find the nearest neighbor in the input image.
        % For that we simply divide position of the pixel (i, j) by 2 (here, 2 is our zoom factor) and round it to nearest full number.
        nn_i = round(i / 2); 
        nn_j = round(j / 2);
        
        % We have to make sure that the index starts from 1 and not 0.
        nn_i = max(nn_i, 1);
        nn_j = max(nn_j, 1);

        % We have to make sure that the index doesn't go out of bound in input image
        nn_i = min(nn_i, row);
        nn_j = min(nn_j, col);
        
        % Copy the desired pixel from input image to output image
        zoomed_img(i, j, :) = img(nn_i, nn_j, :);
    end
end

figure;
imshow(img);
title('Original Image');

figure;
imshow(zoomed_img);
title('2x Zoomed Image');


image=imread('pout. tif');
figure;
imshow(image);
image_double=image
double (image);
[r c]= size(image_double);
cc=input('Enter the value for c=>'); ep=input('Enter the value for gamma=>'); for i=l:r
for j=l:c
imout(i ,j)=CC*power(image_double(i ,j) ,ep);
end
end
figure ,imshow(imout);