### Ex No : 1 Extraction of color components from RGB color image
## Program :
I = imread('lenna.png'); </br>
r = size(I, 1); </br>
c = size(I, 2); </br>
R = zeros(r, c, 3);</br>
G = zeros(r, c, 3);</br>
B = zeros(r, c, 3);</br>
R(:, :, 1) = I(:, :, 1);</br>
G(:, :, 2) = I(:, :, 2);</br>
B(:, :, 3) = I(:, :, 3);</br>
figure, imshow(uint8(R));</br>
figure, imshow(uint8(G));</br>
figure, imshow(uint8(B));</br>
rgbImage = imread('flower.png');</br> 
redChannel = rgbImage(:,:,1); % Red channel </br>
greenChannel = rgbImage(:,:,2); % Green channel </br>
blueChannel = rgbImage(:,:,3); % Blue channel </br>
allBlack = zeros(size(rgbImage, 1), size(rgbImage, 2), 'uint8'); </br>
just_red = cat(3, redChannel, allBlack, allBlack); </br>
just_green = cat(3, allBlack, greenChannel, allBlack); </br>
just_blue = cat(3, allBlack, allBlack, blueChannel); </br>
recombinedRGBImage = cat(3, redChannel, greenChannel, blueChannel); </br>
subplot(3, 3, 2); </br>
imshow(rgbImage); </br>
title('Original RGB Image') </br>
subplot(3, 3, 4); </br>
imshow(just_red); </br>
title('Red Channel in Red') </br>
subplot(3, 3, 5); </br>
imshow(just_green) </br>
title('Green Channel in Green') </br>
subplot(3, 3, 6); </br>
imshow(just_blue); </br>
title('Blue Channel in Blue') </br>
subplot(3, 3, 8); </br>
imshow(recombinedRGBImage); </br>
title('Recombined to Form Original RGB Image Again') </br>

### Ex No : 2 Image enhancement using pixel operation
## Program :
A. Linear Transformation </br>
clc; </br>
clear all; </br>
close all; </br>
pic=imread('grape.jpg'); </br>
subplot(1,2,1) </br>
imshow(pic) </br>
[x,y,z]=size(pic); </br>
if(z==1); </br>
else </br>
pic=rgb2gray(pic); </br>
end </br>
max_gray=max(max(pic)); </br>
max_gray=im2double(max_gray); </br>
pic=im2double(pic); </br>
for i=1:x </br>
for j=1:y </br>
pic_negative(i,j)=max_gray-pic(i,j); </br>
end </br>
end </br>
subplot(1,2,2) </br>
imshow(pic_negative) </br>
B. Logarithmic Transformation </br>
clc; clear all; close all; </br>
f=imread('grape.jpg'); </br>
g=rgb2gray(f); </br>
c=input('Enter the constant value, c = '); </br>
[M,N]=size(g); </br>
for x = 1:M </br>
for y = 1:N </br>
m=double(g(x,y)); </br>
z(x,y)=c.*log10(1+m); </br>
end </br>
end</br>
imshow(f), figure, imshow(z); </br>
C. Power Law Transformation </br>
clear all </br>
close all </br>
RGB=imread('grape.jpg'); </br>
I=rgb2gray(RGB); </br>
I=im2double(I); </br>
[m n] = size(I); </br>
c = 2; </br>
g =[0.5 0.7 0.9 1 2 3 4 5 6]; </br>
for r=1:length(g) </br>
for p = 1 : m </br>
for q = 1 : n </br>
I3(p, q) = c * I(p, q).^ g(r); </br>
end </br>
end </br>
figure, imshow(I3); </br>
title('Power-law transformation'); </br>
xlabel('Gamma='),xlabel(g(r)); </br>
end </br>

### Ex No : 3 Image enhancement using histogram equalization.
## Program :
close all </br>
I = imread('pout.tif'); </br>
imshow(I) </br>
figure, imhist(I) </br>
I2 = histeq(I); </br>
figure, imshow(I2) </br>
figure, imhist(I2) </br>
imwrite (I2, 'pout2.png'); </br>
imfinfo('pout2.png') </br>

### Ex No : 4 Filtering an image using averaging low pass filter in spatial domain and median filter.
## Program :
AVERAGING LOW PASS FILTER </br>
clc </br>
clear all; </br>
close all; </br>
i=imread('grape.jpg'); </br>
a = rgb2gray(i); </br>
b=imnoise(a,'salt & pepper',0.1); </br>
c=imnoise(a,'gaussian'); </br>
d=imnoise(a,'speckle'); </br>
h1=1/9*ones(3,3); </br> 
h2=1/25*ones(5,5); </br>
b1=conv2(b,h1,'same'); </br>
b2=conv2(b,h2,'same');</br>
c1=conv2(c,h1,'same');</br>
c2=conv2(c,h2,'same');</br>
d1=conv2(d,h1,'same');</br>
d2=conv2(d,h2,'same');</br>
figure;</br>
subplot(2,2,1);</br>
imshow(a);</br>
title('original image');</br>
subplot(2,2,2);</br>
imshow(b);</br>
title('Salt & Pepper');</br>
subplot(2,2,3);</br>
imshow(uint8(b1));</br>
title('3X3 Averaging filter');</br>
subplot(2,2,4);</br>
imshow(uint8(b2));</br>
title('5X5 Averaging filter');</br>
figure;</br>
subplot(2,2,1);</br>
imshow(a);</br>
title('original image');</br>
subplot(2,2,2);</br>
imshow(c);</br>
title('Gaussian');</br>
subplot(2,2,3);</br>
imshow(uint8(c1));</br>
title('3X3 Averaging filter');</br>
subplot(2,2,4);</br>
imshow(uint8(c2));</br>
title('5X5 Averaging filter');</br>
figure;</br>
subplot(2,2,1);</br>
imshow(a);</br>
title('original image');</br>
subplot(2,2,2);</br>
imshow(d);</br>
title('Speckle');</br>
subplot(2,2,3);</br>
imshow(uint8(d1));</br>
title('3X3 Averaging filter');</br>
subplot(2,2,4);</br>
imshow(uint8(d2));</br>
title('5X5 Averaging filter');</br>
MEDIAN FILTER</br>
clc;</br>
clear all;</br>
close all;</br>
a = imread('grape.jpg');</br>
I = rgb2gray(a);</br>
J = imnoise(I,'salt & pepper',0.02);</br>
K = medfilt2(J);</br>
figure;</br>
subplot(1,3,1);</br>
imshow(I);</br>
title('Original image');</br>
subplot(1,3,2)</br>
imshow(J);</br>
title('Noisy image');</br>
subplot(1,3,3);</br>
imshow(K);</br>
title('Median filtered image');</br>

### Ex No : 5 Sharpen an image using 2-D laplacian high pass filter in spatial domain.
## Program :
i = imread("grape.jpg");</br>
subplot(2,2,1);</br>
a =imshow("grape.jpg");</br>
title("Original image");</br>
a= rgb2gray(i);</br>
Lap=[0 1 0; 1 -4 1; 0 1 0];</br>
a1 = conv2(a,Lap,'same');</br>
a2 = uint8(a1);</br>
subplot(2,2,2);</br>
imshow(abs(a-a2),[])</br>
title("Laplacian filtered image");</br>
lap=[-1 -1 -1; -1 8 -1; -1 -1 -1];</br>
a3=conv2(a,lap,'same');</br>
a4=uint8(a3);</br>
subplot(2,2,3);</br>
imshow(abs(a+a4),[])</br>
title("High boost filtered image");</br>

### Ex No : 6 Smoothing of an image using low pass filter and high pass filter in frequency domain (Butterworth LPF and HPF)
## Program :
% MATLAB Code | Butterworth Low Pass Filter</br>
clc;</br>
clear all;</br>
close all;</br>
a = imread("grape.jpg");</br>
input_image = rgb2gray(a);</br>
[M, N] = size(input_image);</br>
FT_img = fft2(double(input_image));> 2; </br>
D0 = 20;</br>
u = 0:(M-1);</br>
v = 0:(N-1);</br>
idx = find(u > M/2);</br>
u(idx) = u(idx) - M;</br>
idy = find(v > N/2);</br>
v(idy) = v(idy) - N;</br>
[V, U] = meshgrid(v, u);</br>
D = sqrt(U.^2 + V.^2);</br>
H = 1./(1 + (D./D0).^(2*n))</br>
G = double(H).*double(FT_img);</br>
output_image = real(ifft2(double(G)));</br>
subplot(2, 1, 1), imshow(input_image),</br>
title("Original Image");</br>
subplot(2, 1, 2), imshow(output_image, [ ]);</br>
title("Butterworth lowpass filtered Image");</br>
% MATLAB Code | Butterworth High Pass Filter</br>
a = imread("grape.jpg");</br>
input_image= rgb2gray(a);</br>
[M, N] = size(input_image);</br>
FT_img = fft2(double(input_image));</br>
n = 2;</br>
D0 = 10;</br>
u = 0:(M-1);</br>
v = 0:(N-1);</br>
idx = find(u > M/2);</br>
u(idx) = u(idx) - M;</br>
idy = find(v > N/2);</br>
v(idy) = v(idy) - N;</br>
[V, U] = meshgrid(v, u);</br>
D = sqrt(U.^2 + V.^2);</br>
H = 1./(1 + (D0./D).^(2*n));</br>
G = H.*FT_img;</br>
output_image = real(ifft2(double(G)));</br>
subplot(2, 1, 1), imshow(input_image),</br>
title("Original Image");</br>
subplot(2, 1, 2), imshow(output_image, [ ]);</br>
title("Butterworth highpass filtered Image");</br>
### Ex No : 7 Program for morphological image operations-erosion, dilation, opening & closing
## Program :
% Morphological image operations - Erosion</br>
originalBW = imread('cameraman.tif');</br>
se = strel('line',5,40);</br>
erodedBW = imerode(originalBW,se);</br>
figure, imshow(originalBW);</br>
title("Original Image");</br>
figure, imshow(erodedBW)</br>
title("Eroded Image");</br>
% Morphological image operations - Dilation</br>
originalBW = imread('text.png');</br>
se = strel('line',9,50);</br>
dilatedBW = imdilate(originalBW,se);</br>
figure, imshow(originalBW),</br>
title("Original Image");</br>
figure, imshow(dilatedBW)</br>
title("Dilated Image");</br>
% Morphological image operations - Opening</br>
original = imread('cameraman.tif');</br>
se = strel('disk',3);</br>
afterOpening = imopen(original,se);</br>
figure, imshow(original),</br>
title("Original Image");</br>
figure, imshow(afterOpening,[])</br>
title("Image after opening");</br>
% Morphological image operations - Closing</br>
originalBW = imread('circles.png');</br>
figure, imshow(originalBW);</br>
title("Original Image");</br>
se = strel('disk',6);</br>
closeBW = imclose(originalBW,se);</br>
figure, imshow(closeBW);</br>
title("Image after closing");</br>

### Ex No : 9 Program for image compression using Huffman coding
## Program :
symbols = 1:6;</br>
p = [.5 .125 .125 .125 .0625 .0625];</br>
dict = huffmandict(symbols,p);</br>
inputSig = randsrc(100,1,[symbols;p]);</br>
code = huffmanenco(inputSig,dict);</br>
sig = huffmandeco(code,dict);</br>
isequal(inputSig,sig)</br>
binarySig = de2bi(inputSig);</br>
seqLen = numel(binarySig)</br>
binaryComp = de2bi(code);</br>
encodedLen = numel(binaryComp)</br>
inputSig = {'a2',44,'a3',55,'a1'}</br>
dict = {'a1',0; 'a2',[1,0]; 'a3',[1,1,0]; 44,[1,1,1,0]; 55,[1,1,1,1]}</br>
enco = huffmanenco(inputSig,dict);</br>
sig = huffmandeco(enco,dict)</br>
isequal(inputSig,sig)</br>

### Exp: 10. Pattern Classification Methods.
## PROGRAM:
[x,t] = iris_dataset;</br>
net = patternnet(10);</br>
net = train(net,x,t);</br>
view(net)</br>
y = net(x);</br>
perf = perform(net,t,y);</br>
classes = vec2ind(y);</br>
