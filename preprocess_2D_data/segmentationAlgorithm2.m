function new_mask = segmentationAlgorithm2(x)
CC = bwconncomp(x > 0);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
mask = x*0;
mask(CC.PixelIdxList{idx}) = 1;
fmask = flip(mask, 2);

symmask = (mask + flip(mask, 2)) > 0;
se = strel('disk',3);

dmask = imerode(symmask, se);
CC = bwconncomp(dmask);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
new_mask = dmask * 0;
new_mask(CC.PixelIdxList{idx}) = 1;

se5 = strel('disk',5);
% reverse dilation
new_mask = imdilate(new_mask, se5);
new_mask = imfill(new_mask, 'holes');
end 