function masked = applyMask(x, mask)
masked = x;
masked(mask == 0) = 0;
end
