%get directory for files separated by their sets (Train, Test, Validation)
source_dir = '/dcs05/ciprian/smart/pocus/rushil/full_png';

% Get a list of all .png files in separated_masked_png
sep_png_files = dir(fullfile(source_dir, '*', '*', '*.png'));

%get masks for each clip for all patients
mask_dir = '/dcs05/ciprian/smart/pocus/data/mask/';

%set following directories
output_rescaled_dir = '/dcs05/ciprian/smart/pocus/rushil/bounding_box_rescaled/';
output_rectilinear_dir = '/dcs05/ciprian/smart/pocus/rushil/rectilinear_rescaled/';
output_masked_dir = '/dcs05/ciprian/smart/pocus/rushil/masked/';


% Iterate through each .png file in source_dir
for k = 1:length(sep_png_files)
    % Current png file
    slice_file = sep_png_files(k).name;
    slice_folder = sep_png_files(k).folder;

    % Extract patient_id and DICOM name (ex: PID and IM000001)
    [~, filename, ~] = fileparts(slice_file);  
    parts = strsplit(filename, '_');
    patient_id = parts{1};  % Extract PID
    dicom_filename = parts{2};  % Extract DICOM name

    % Find the corresponding mask file using PID and DICOM name
    mask_file = fullfile(mask_dir, patient_id, [dicom_filename '.png']);

    % Extract the subfolder structure (training/b-line/etc.) from the current file
    split_parts = strsplit(slice_folder, filesep);
    data_set_split = split_parts{end-1};  % 'training', 'testing', 'validation'
    data_label_category = split_parts{end};      % 'b-line', 'control'

    % Define the new output names and paths for rescaled and rectilinear images
    output_rescaled_file = fullfile(output_rescaled_dir, data_set_split, data_label_category, slice_file);
    rect_output_file = fullfile(output_rectilinear_dir, data_set_split, data_label_category, slice_file);
    masked_output_file = fullfile(output_masked_dir, data_set_split, data_label_category, slice_file);


    output_rescaled_subdir = fullfile(output_rescaled_dir, data_set_split, data_label_category);
    output_rectilinear_subdir = fullfile(output_rectilinear_dir, data_set_split, data_label_category);
    output_masked_subdir = fullfile(output_masked_dir, data_set_split, data_label_category);

    
    % Make directories if they do not exist
    if ~exist(output_rescaled_subdir, 'dir')
        mkdir(output_rescaled_subdir);
    end
    if ~exist(output_rectilinear_subdir, 'dir')
        mkdir(output_rectilinear_subdir);
    end
    if ~exist(output_masked_subdir, 'dir')
        mkdir(output_masked_subdir);
    end

    % The rest of the processing remains the same for each image
    image = imread(fullfile(slice_folder, slice_file));
    new_mask = imread(mask_file);

    % Apply clip level mask
    image = applyMask(image, new_mask);
    target_size = [224, 224];
    masked_resize = imresize(image, target_size);
    imwrite(masked_resize, masked_output_file);


    % Bounding box and cropping logic
    binary_mask = new_mask > 1;
    props = regionprops(binary_mask, 'BoundingBox');
    boundingBox = props.BoundingBox;

    x = boundingBox(1);
    y = boundingBox(2);
    width = boundingBox(3);
    height = boundingBox(4);

    % Get box dimensions
    verticesX = [x, x, x+width, x+width, x];
    verticesY = [y+height, y, y, y+height, y+height];

    verticesX = round(verticesX);
    verticesY = round(verticesY);

    cropped_image = image(verticesY(2):verticesY(1), verticesX(1):verticesX(3), :);

    % Resize and save the cropped image
    resized_image = imresize(cropped_image, target_size);
    imwrite(resized_image, output_rescaled_file);

    % Polar conversion and rectilinear image creation
    apex = [x + width/2, y];
    [theta, rho] = meshgrid(linspace(0, pi, width), linspace(0, height, height));
    [X, Y] = pol2cart(theta, rho);
    X = X + apex(1);
    Y = Y + apex(2);
    X = min(max(X, 1), size(image, 2));
    Y = min(max(Y, 1), size(image, 1));

    image = double(im2gray(image));
    rectilinear_image = interp2(image, X, Y, 'nearest', 0);
    edges = edge(rectilinear_image, 'Roberts');
    [yCoords, xCoords] = find(edges);
    rectilinear_image = rectilinear_image(min(yCoords):max(yCoords), min(xCoords):max(xCoords));
    rectilinear_image = imresize(rectilinear_image, [224 224]);
    rectilinear_image = fliplr(rectilinear_image);
    rectilinear_image = uint8(rectilinear_image);

    % Save the rectilinear image
    imwrite(rectilinear_image, rect_output_file);
end