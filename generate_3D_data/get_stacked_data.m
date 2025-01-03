csvFile = readtable('/dcs05/ciprian/smart/pocus/data/dicom_labels_2d_with_labels.csv');

for i=1:size(csvFile(:,1),1)
    csvFile{i,1} = {strcat('/dcs05/ciprian/smart/pocus/rushil/bounding_box_rescaled/', csvFile{i,6}, '/', csvFile{i,5}, '/', csvFile{i,2}, '_', csvFile{i,3}, '_',sprintf("%03d", csvFile{i,4}), '.png')};
end

uniquePID = unique(csvFile.id_patient);

for i=1:size(uniquePID,1)
    currentPID = uniquePID(i);
    currentPID_table = csvFile(strcmp(csvFile.id_patient, currentPID), :);
    uniqueClip = unique(currentPID_table.id_screenshot);
    for j=1:size(uniqueClip, 1)
        currentClip = uniqueClip(j);
        currentClip_table = currentPID_table(strcmp(currentPID_table.id_screenshot, currentClip), :);
        numSlices = size(currentClip_table,1);
        imageStack = zeros(256, 384, 3, numSlices);
        for k = 1:numSlices
            image = imread(currentClip_table.file{k});
            image = imresize(image, [256 384]);
            image = repmat(image, [1, 1, 3]);
            imageStack(:, :, :, k) = image; % 4th dimension is the slice index
        end
        % Split or pad the imageStack to 45 slices per clip
        totalClips = ceil(numSlices / 45);
        for clipNum = 1:totalClips
            startIdx = (clipNum - 1) * 45 + 1;
            endIdx = min(clipNum * 45, numSlices);
            currentStack = imageStack(:, :, :, startIdx:endIdx);

            % If less than 45 slices, pad with the last slice
            if size(currentStack, 4) < 45
                padSize = 45 - size(currentStack, 4);
                lastSlice = currentStack(:, :, :, end); % Last available slice
                padStack = repmat(lastSlice, 1, 1, 1, padSize);
                currentStack = cat(4, currentStack, padStack);
            end

            % Permute to (n_slices, H, W, 3)
            currentStack = permute(currentStack, [4, 1, 2, 3]);

            % Determine label
            if any(strcmp(currentClip_table.label, 'b-line'))
                label = 'b-line';
            else
                label = 'control';
            end

            % Determine split
            uniqueLabels = unique(currentClip_table.group_split);
            if length(uniqueLabels) == 1
                split = uniqueLabels{1};
            else
                disp([currentPID currentClip])
            end

            % Save with a numbered filename
            fileName = strcat('/dcs05/ciprian/smart/pocus/rushil/bounding_box_stacked_data/', split, '/', label, '/', currentPID{1}, '_', currentClip{1}, '_', num2str(clipNum), '.mat');
            save(fileName, 'currentStack');
        end
    end
end
