%read input files
csvData = readtable("/dcs05/ciprian/smart/pocus/rushil/JHU_SCANS.csv");
folderList = dir('/Users/rushil/Downloads/NEW JHU SEPSIS 9-16 UPLOAD');
folderList = folderList(~ismember({folderList.name}, {'.', '..', '.DS_Store'}));

%create random splits
numPatients = numel(folderList);
numTrain = round(0.7 * numPatients);
numTest = round(0.2 * numPatients);
numVal = numPatients - numTrain - numTest;

% Shuffle the folderList to randomize the split
shuffledIndices = randperm(numPatients);

% Indices for training, testing, and validation
trainIdx = shuffledIndices(1:numTrain);
testIdx = shuffledIndices(numTrain+1:numTrain+numTest);
valIdx = shuffledIndices(numTrain+numTest+1:end);

%define output directory
full_png = '/dcs05/ciprian/smart/pocus/rushil/full_png';

for i = 1:numel(folderList)
    %get PID and properly formatted name
    currentFolder = fullfile(folderList(i).folder, folderList(i).name);
    originalName = folderList(i).name;
    parts = strsplit(originalName, ' ');
    filename = parts{1};
    datePart = parts{2};
    dateNumbers = sscanf(datePart, '%d-%d-%d');
    formattedDate = sprintf('%02d%02d%02d', dateNumbers(1), dateNumbers(2), dateNumbers(3));
    formattedName = [filename '_' formattedDate];
    dicomFiles = dir(fullfile(currentFolder, "/IM*"));
    %check set of patient clip
    if ismember(i, trainIdx)
        setType = 'training';
    elseif ismember(i, testIdx)
        setType = 'test';
    else
        setType = 'validation';
    end

    for j = 1:numel(dicomFiles)
        currentFile = fullfile(currentFolder, dicomFiles(j).name);
        C = dicomread(currentFile);

        patientMatch = strcmp(csvData.Patient, formattedName);
        clipMatch = csvData.Clip == j;
        matchedRow = find(patientMatch & clipMatch);
        %get label from csv
        category = csvData.Diagnosis{matchedRow};
       
        %skipping if the clip is Undiagnostic
        if strcmp(category, 'Undiagnostic')
            fprintf('Skipping %s clip %d due to Undiagnostic category.\n', formattedName, j);
            continue;
        end
        %create output directory
        outputFolder = fullfile(full_png, setType, category);
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end

        for k = 1:size(C, 4)
            %load slice
            im_new = C(:, :, 1, k);
            name = strcat(formattedName, "_", dicomFiles(j).name, "_", sprintf('%03d', k), ".png");
            fullFileName = fullfile(outputFolder, name);
            %save slice
            imwrite(im_new, fullFileName);
        end
    end
end