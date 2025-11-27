package fr.ird.plantsam.service;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.io.FileUtils;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.core.io.FileSystemResource;
import org.apache.commons.io.FilenameUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.FileSystemUtils;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

@Service
public class FileService {

    private Map<String, FileGroup> fileGroups;
    private Path storagePath = Paths.get("uploads");
    private Path processedPath =  Paths.get("processed");
    private Path segmentedPath = Paths.get("segmented");

    public FileService(
            @Value("${file.upload-dir:uploads}") String uploadDir,
            @Value("${file.processed-dir:processed}") String processedDir,
            @Value("${file.segmented-dir:segmented}") String segmentedDir) {
        this.storagePath = Paths.get(uploadDir).toAbsolutePath().normalize();
        this.processedPath = Paths.get(processedDir).toAbsolutePath().normalize();
        this.segmentedPath = Paths.get(segmentedDir).toAbsolutePath().normalize();
        fileGroups = new HashMap<>();

        try {
            Files.createDirectories(storagePath);
            Files.createDirectories(processedPath);
            Files.createDirectories(segmentedPath);
            
            // Reconstruct fileGroups from existing disk folders
            reconstructFileGroupsFromDisk();
        } catch (IOException e) {
            throw new RuntimeException("Could not create upload directories", e);
        }
    }
    
    /**
     * Reconstructs the fileGroups HashMap from existing folders on disk.
     * This is called on service initialization to restore projects after server restart.
     */
    private void reconstructFileGroupsFromDisk() throws IOException {
        System.out.println("🔄 Reconstructing file groups from disk...");
        
        // Scan uploads directory for group folders
        if (!Files.exists(storagePath)) {
            System.out.println("⚠️ No uploads directory found, skipping reconstruction");
            return;
        }
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(storagePath)) {
            for (Path groupDir : stream) {
                if (!Files.isDirectory(groupDir)) continue;
                
                String groupId = groupDir.getFileName().toString();
                
                // Find corresponding processed and segmented directories
                Path processedDir = processedPath.resolve(groupId);
                
                // Get files from the upload directory
                List<Path> uploadedFiles = new ArrayList<>();
                try (DirectoryStream<Path> fileStream = Files.newDirectoryStream(groupDir)) {
                    for (Path file : fileStream) {
                        if (Files.isRegularFile(file)) {
                            // Skip metadata files
                            String fileName = file.getFileName().toString();
                            if (!fileName.startsWith(".")) {
                                uploadedFiles.add(file);
                            }
                        }
                    }
                }
                
                if (uploadedFiles.isEmpty()) {
                    System.out.println("⚠️ Skipping empty group: " + groupId);
                    continue;
                }
                
                // Sort files by their index suffix (e.g., image_0.png, image_1.png)
                uploadedFiles.sort((a, b) -> {
                    String nameA = a.getFileName().toString();
                    String nameB = b.getFileName().toString();
                    
                    // Extract index from filename (format: basename_index.ext)
                    int indexA = extractFileIndex(nameA);
                    int indexB = extractFileIndex(nameB);
                    
                    return Integer.compare(indexA, indexB);
                });
                
                // Determine project name from metadata file or segmented folders
                String projectName = loadProjectMetadata(groupId);
                if (projectName == null) {
                    projectName = findProjectNameForGroup(groupId);
                }
                if (projectName == null) {
                    // Fallback: use groupId as project name if no metadata or segmented folder found
                    projectName = "project_" + groupId.substring(0, 8);
                }
                
                Path segmentedDir = segmentedPath.resolve(projectName);
                
                // Create FileGroup
                FileGroup fileGroup = new FileGroup(projectName);
                fileGroup.setUploadDirPath(groupDir);
                fileGroup.setProcessedDirPath(processedDir);
                fileGroup.setSegmentedDirPath(segmentedDir);
                
                // Add all files to the group
                for (int i = 0; i < uploadedFiles.size(); i++) {
                    Path file = uploadedFiles.get(i);
                    fileGroup.addOriginalFile(i, file.toString());
                    
                    // Check if any processed file exists with various possible prefixes
                    String originalFileName = file.getFileName().toString();
                    Path processedFile = findProcessedFile(processedDir, originalFileName);
                    if (processedFile != null) {
                        fileGroup.addProcessedFile(i, processedFile.toString());
                    }
                    
                    // Check if segmented file exists
                    String baseName = FilenameUtils.removeExtension(originalFileName);
                    String segmentedFileName = baseName + "_mask.png";
                    Path segmentedFile = segmentedDir.resolve(segmentedFileName);
                    if (Files.exists(segmentedFile)) {
                        fileGroup.addSegmentedFile(i, segmentedFile.toString());
                    }
                }
                
                fileGroups.put(groupId, fileGroup);
                System.out.println("✅ Reconstructed group: " + groupId + " (" + projectName + ") with " + uploadedFiles.size() + " files");
            }
        }
        
        System.out.println("✅ Reconstruction complete. Total groups: " + fileGroups.size());
    }
    
    /**
     * Finds a processed file that corresponds to the original filename.
     * Checks various possible prefixes used during processing.
     */
    private Path findProcessedFile(Path processedDir, String originalFileName) throws IOException {
        if (!Files.exists(processedDir)) return null;
        
        // Common prefixes used during image processing
        String[] prefixes = {
            "processed_",
            "cleared_points_",
            "cleared_rectangles_",
            "segmented_with_points_",
            "union_segmented_",
            "intersection_segmented_",
            "negative_point_",
            "positive_",
            "negative_"
        };
        
        // Try each prefix
        for (String prefix : prefixes) {
            Path candidateFile = processedDir.resolve(prefix + originalFileName);
            if (Files.exists(candidateFile)) {
                return candidateFile;
            }
        }
        
        // Also check for files that end with the original filename (for step files)
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(processedDir)) {
            for (Path file : stream) {
                String fileName = file.getFileName().toString();
                if (fileName.endsWith(originalFileName)) {
                    return file;
                }
            }
        }
        
        return null;
    }
    
    /**
     * Extracts the file index from a filename with format: basename_index.ext
     */
    private int extractFileIndex(String fileName) {
        String nameWithoutExt = FilenameUtils.removeExtension(fileName);
        int lastUnderscore = nameWithoutExt.lastIndexOf('_');
        if (lastUnderscore != -1) {
            try {
                return Integer.parseInt(nameWithoutExt.substring(lastUnderscore + 1));
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }
    
    /**
     * Finds the project name by checking which segmented folder corresponds to this groupId.
     * This is done by checking if any processed files from this group exist in segmented folders.
     */
    private String findProjectNameForGroup(String groupId) throws IOException {
        if (!Files.exists(segmentedPath)) return null;
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(segmentedPath)) {
            for (Path projectDir : stream) {
                if (!Files.isDirectory(projectDir)) continue;
                
                // Check if this project folder has any mask files
                // We'll try to match by checking if any files exist
                try (DirectoryStream<Path> maskStream = Files.newDirectoryStream(projectDir, "*.png")) {
                    if (maskStream.iterator().hasNext()) {
                        // Found a project with masks, return its name
                        return projectDir.getFileName().toString();
                    }
                }
            }
        }
        
        return null;
    }

    public String addGroup(String groupName) {
        String groupId = UUID.randomUUID().toString();
        FileGroup newGroup = new FileGroup(groupName);
        try {
            Path groupUploadDir = storagePath.resolve(groupId);
            Path groupProcessedDir = processedPath.resolve(groupId);
            // Use the actual project name for the segmented folder
            Path groupSegmentedDir = segmentedPath.resolve(groupName);
            Files.createDirectories(groupUploadDir);
            Files.createDirectories(groupProcessedDir);
            Files.createDirectories(groupSegmentedDir);

            newGroup.setUploadDirPath(groupUploadDir);
            newGroup.setProcessedDirPath(groupProcessedDir);
            newGroup.setSegmentedDirPath(groupSegmentedDir);
            
            // Save project metadata to disk for persistence across restarts
            saveProjectMetadata(groupId, groupName);
        } catch (Exception e) {
            throw new RuntimeException("Could not create upload directories", e);
        }
        fileGroups.put(groupId, newGroup);
        return groupId;
    }
    
    /**
     * Saves project metadata (groupId -> projectName mapping) to disk
     */
    private void saveProjectMetadata(String groupId, String projectName) throws IOException {
        Path metadataFile = storagePath.resolve(groupId).resolve(".metadata");
        Files.writeString(metadataFile, projectName, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
    
    /**
     * Loads project name from metadata file
     */
    private String loadProjectMetadata(String groupId) {
        try {
            Path metadataFile = storagePath.resolve(groupId).resolve(".metadata");
            if (Files.exists(metadataFile)) {
                return Files.readString(metadataFile).trim();
            }
        } catch (IOException e) {
            System.err.println("⚠️ Error reading metadata for group " + groupId + ": " + e.getMessage());
        }
        return null;
    }

    public void addFile(String groupId, MultipartFile file) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            return;
        }

        String originalFileName = file.getOriginalFilename();
        String fileExtension = (originalFileName != null && originalFileName.contains("."))
                ? originalFileName.substring(originalFileName.lastIndexOf("."))
                : ".bin";

        int fileIndex = group.getFilesCount();
        String fileBaseName = FilenameUtils.removeExtension(originalFileName);
        String storedFileName = fileBaseName + "_" + fileIndex + fileExtension;
        Path destinationFile = group.getUploadDirPath().resolve(storedFileName);

        file.transferTo(destinationFile.toFile());
        group.addOriginalFile(fileIndex, destinationFile.toString());
    }


    public byte[] getOriginalImage(String groupId, int fileIndex) {
        try {
            FileGroup group = fileGroups.get(groupId);
            if (group == null) { return null; };

            String filePath = group.getOriginalFilePath(fileIndex);
            if (filePath == null) { return null; };

            return Files.readAllBytes(Paths.get(filePath));
        } catch (IOException e) {
            return null;
        }
    }

    public int getFileCount(String groupId) {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            return 0;
        }
        return group.getFilesCount();
    }

    // Helper method to save final mask to segmented folder
    private void saveFinalMask(FileGroup group, int fileIndex, byte[] maskData, String originalFilePath) throws IOException {
        if (group.getSegmentedDirPath() == null) return;
        
        String fileName = Paths.get(originalFilePath).getFileName().toString();
        String baseName = FilenameUtils.removeExtension(fileName);
        String segmentedFileName = baseName + "_mask.png";
        Path segmentedFile = group.getSegmentedDirPath().resolve(segmentedFileName);
        
        Files.write(segmentedFile, maskData, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        group.addSegmentedFile(fileIndex, segmentedFile.toString());
        
        System.out.println("✅ Final mask saved to segmented folder: " + segmentedFile);
    }

    public boolean processImage(String groupId, int fileIndex) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<byte[]> response = restTemplate.exchange(
                "http://localhost:8000/process",
                HttpMethod.POST,
                requestEntity,
                byte[].class
        );

        if (response.getStatusCode() == HttpStatus.OK) {
            String processedFileName = "processed_" + Paths.get(originalFilePath).getFileName();
            Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
            Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

            group.addProcessedFile(fileIndex, processedFile.toString());
            
            // Save final mask to segmented folder
            saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
            
            return true;
        }
        return false;
    }

    public boolean setPoint(String groupId, int fileIndex, int x, int y, boolean positive) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Coordonnées reçues du frontend: x=" + x + ", y=" + y);
        System.out.println("Fichier à envoyer: " + originalFilePath);

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("x", String.valueOf(x));
        body.add("y", String.valueOf(y));

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = positive
                ? "http://localhost:8000/positive_point"
                : "http://localhost:8000/negative_point";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                String suffix = positive ? "positive" : "negative";
                String processedFileName = suffix + "_processed_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }


    public byte[] getProcessedImage(String groupId, int fileIndex) {
        try {
            FileGroup group = fileGroups.get(groupId);
            if (group == null) {
                return null;
            }

            String filePath = group.getProcessedFilePath(fileIndex);
            if (filePath == null) {
                return null;
            }

            return Files.readAllBytes(Paths.get(filePath));
        } catch (IOException e) {
            return null;
        }
    }

    public boolean deleteGroup(String groupId) {
        try {
            FileGroup group = fileGroups.get(groupId);
            
            // Delete uploads folder (organized by groupId)
            FileSystemUtils.deleteRecursively(storagePath.resolve(groupId));
            
            // Delete processed folder (organized by groupId)
            FileSystemUtils.deleteRecursively(processedPath.resolve(groupId));
            
            // Delete segmented folder (organized by project name)
            if (group != null && group.getSegmentedDirPath() != null) {
                FileSystemUtils.deleteRecursively(group.getSegmentedDirPath());
                System.out.println("✅ Deleted segmented folder: " + group.getSegmentedDirPath());
            }
            
            fileGroups.remove(groupId);
            System.out.println("✅ Successfully deleted all folders for group: " + groupId);
            return true;
        } catch (IOException e) {
            System.err.println("❌ Error deleting group folders: " + e.getMessage());
            return false;
        }
    }

    public boolean deleteFile(String groupId, int fileIndex) {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            return false;
        }
        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) {
            return false;
        }
        try {
            Files.deleteIfExists(Paths.get(originalFilePath));
        } catch (IOException e) {
            return false;
        }

        String processedFilePath = group.getProcessedFilePath(fileIndex);
        if (processedFilePath == null) {
            return false;
        }
        try {
            Files.deleteIfExists(Paths.get(processedFilePath));
        } catch (IOException e) {
            return false;
        }

        group.originalFiles.remove(fileIndex);
        group.processedFiles.remove(fileIndex);
        return true;
    }

    public boolean segmentWithPoints(String groupId, int fileIndex, String positivePoints, String negativePoints, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Segment avec points - Positifs: " + positivePoints + ", Négatifs: " + negativePoints + ", StartType: " + startType);

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("positive_points", positivePoints);
        body.add("negative_points", negativePoints);
        body.add("start_type", startType);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = "http://localhost:8000/segment_with_points";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "segmented_with_points_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }

    public boolean segmentUnion(String groupId, int fileIndex, int x, int y, int pointCount, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;


        System.out.println("Segment union - Point: (" + x + ", " + y + "), Count: " + pointCount + ", StartType: " + startType);

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("x", String.valueOf(x));
        body.add("y", String.valueOf(y));
        body.add("point_count", String.valueOf(pointCount));
        body.add("start_type", startType);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = "http://localhost:8000/segment_union";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                System.out.println("Segment union: succès, image traitée sauvegardée");
                String processedFileName = "union_segmented_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            } else {
                System.out.println("Segment union: échec, statut HTTP: " + response.getStatusCode());
                return false;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }

    public boolean segmentIntersection(String groupId, int fileIndex, int x, int y, int pointCount, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Segment intersection - Point: (" + x + ", " + y + "), Count: " + pointCount + ", StartType: " + startType);

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("x", String.valueOf(x));
        body.add("y", String.valueOf(y));
        body.add("point_count", String.valueOf(pointCount));
        body.add("start_type", startType);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = "http://localhost:8000/segment_intersection";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "intersection_segmented_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }

    public boolean clearPoints(String groupId, int fileIndex) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = "http://localhost:8000/clear_points";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "cleared_points_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }

    public boolean saveNegativePointResult(String groupId, int fileIndex, byte[] imageData) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        String processedFileName = "negative_point_" + Paths.get(originalFilePath).getFileName();
        Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
        Files.write(processedFile, imageData, StandardOpenOption.CREATE);

        group.addProcessedFile(fileIndex, processedFile.toString());
        return true;
    }

    public boolean saveStepImage(String groupId, int fileIndex, byte[] imageData, String stepName) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        String stepFileName = stepName + "_" + Paths.get(originalFilePath).getFileName();
        Path stepFile = group.getProcessedDirPath().resolve(stepFileName);

        System.out.println("Sauvegarde de l'étape: " + stepFile.toAbsolutePath());

        Files.write(stepFile, imageData, StandardOpenOption.CREATE);
        group.addProcessedFile(fileIndex, stepFile.toString());
        return true;
    }

    public byte[] getStepImage(String groupId, int fileIndex, String stepName) {
        try {
            FileGroup group = fileGroups.get(groupId);
            if (group == null) return null;

            String originalFilePath = group.getOriginalFilePath(fileIndex);
            if (originalFilePath == null) return null;

            String stepFileName = stepName + "_" + Paths.get(originalFilePath).getFileName();
            Path stepFile = group.getProcessedDirPath().resolve(stepFileName);

            if (Files.exists(stepFile)) {
                return Files.readAllBytes(stepFile);
            }
            return null;
        } catch (IOException e) {
            return null;
        }
    }

    public boolean removeRectangle(String groupId, int fileIndex, int x, int y, int width, int height, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Remove rectangle - Coords: (" + x + ", " + y + "), Size: " + width + "x" + height + ", StartType: " + startType);

        RestTemplate restTemplate = new RestTemplate();

        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("x", String.valueOf(x));
        body.add("y", String.valueOf(y));
        body.add("width", String.valueOf(width));
        body.add("height", String.valueOf(height));
        body.add("start_type", startType);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = "http://localhost:8000/remove_rectangle";

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "rectangle_removed_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python pour remove_rectangle: " + e.getMessage());
        }
        return false;
    }

    public boolean clearRectangles(String groupId, int fileIndex) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            System.out.println("Group not found: " + groupId);
            return false;
        }

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) {
            System.out.println("Original file path not found for group: " + groupId + ", file: " + fileIndex);
            return false;
        }

        System.out.println("Clearing rectangles for group: " + groupId + ", file: " + fileIndex);

        RestTemplate restTemplate = new RestTemplate();

        try {
            FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", fileResource);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<String> response = restTemplate.exchange(
                    "http://localhost:8000/clear_rectangles",
                    HttpMethod.POST,
                    requestEntity,
                    String.class
            );

            if (response.getStatusCode() == HttpStatus.OK) {
                System.out.println("Rectangles cleared successfully for group: " + groupId + ", file: " + fileIndex);

                // Optionnel : sauvegarder l'état après nettoyage des rectangles
                String processedFileName = "cleared_rectangles_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);

                // Récupérer l'image actuelle après nettoyage
                byte[] currentImageData = getProcessedImage(groupId, fileIndex);
                if (currentImageData != null) {
                    Files.write(processedFile, currentImageData, StandardOpenOption.CREATE);
                    group.addProcessedFile(fileIndex, processedFile.toString());
                }

                return true;
            } else {
                System.out.println("Failed to clear rectangles, HTTP status: " + response.getStatusCode());
                return false;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API to clear rectangles: " + e.getMessage());
            return false;
        }
    }

    public boolean applyUnion(String groupId, int fileIndex, byte[] previousMask) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Applying union algorithm between masks");

        RestTemplate restTemplate = new RestTemplate();

        // Create the file resource for the original image
        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        
        // Create a temporary file for the previous mask
        Path tempPreviousMask = Files.createTempFile("previous_mask_", ".png");
        Files.write(tempPreviousMask, previousMask);
        FileSystemResource previousMaskResource = new FileSystemResource(tempPreviousMask.toFile());

        // Set up the multipart request
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("previous_mask", previousMaskResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    "http://localhost:8000/apply_union",
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            // Clean up the temporary file
            Files.deleteIfExists(tempPreviousMask);

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "processed_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API for union: " + e.getMessage());
            // Clean up the temporary file in case of error
            Files.deleteIfExists(tempPreviousMask);
        }
        return false;
    }

    public boolean applyIntersection(String groupId, int fileIndex, byte[] previousMask) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Applying intersection algorithm between masks");

        RestTemplate restTemplate = new RestTemplate();

        // Create the file resource for the original image
        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        
        // Create a temporary file for the previous mask
        Path tempPreviousMask = Files.createTempFile("previous_mask_", ".png");
        Files.write(tempPreviousMask, previousMask);
        FileSystemResource previousMaskResource = new FileSystemResource(tempPreviousMask.toFile());

        // Set up the multipart request
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("previous_mask", previousMaskResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    "http://localhost:8000/apply_intersection",
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            // Clean up the temporary file
            Files.deleteIfExists(tempPreviousMask);

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "processed_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API for intersection: " + e.getMessage());
            // Clean up the temporary file in case of error
            Files.deleteIfExists(tempPreviousMask);
        }
        return false;
    }

    public boolean applyIou(String groupId, int fileIndex, byte[] previousMask) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Applying IoU algorithm between masks");

        RestTemplate restTemplate = new RestTemplate();

        // Create the file resource for the original image
        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        
        // Create a temporary file for the previous mask
        Path tempPreviousMask = Files.createTempFile("previous_mask_", ".png");
        Files.write(tempPreviousMask, previousMask);
        FileSystemResource previousMaskResource = new FileSystemResource(tempPreviousMask.toFile());

        // Set up the multipart request
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("previous_mask", previousMaskResource);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    "http://localhost:8000/apply_iou",
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            // Clean up the temporary file
            Files.deleteIfExists(tempPreviousMask);

            if (response.getStatusCode() == HttpStatus.OK) {
                String processedFileName = "processed_" + Paths.get(originalFilePath).getFileName();
                Path processedFile = group.getProcessedDirPath().resolve(processedFileName);
                Files.write(processedFile, response.getBody(), StandardOpenOption.CREATE);

                group.addProcessedFile(fileIndex, processedFile.toString());
                
                // Save final mask to segmented folder
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API for IoU: " + e.getMessage());
            // Clean up the temporary file in case of error
            Files.deleteIfExists(tempPreviousMask);
        }
        return false;
    }

    private static class FileGroup {
        @Getter String groupName;
        private final Map<Integer, String> originalFiles = new HashMap<>();
        private final Map<Integer, String> processedFiles = new HashMap<>();
        private final Map<Integer, String> segmentedFiles = new HashMap<>();
        @Getter @Setter private Path uploadDirPath;
        @Getter @Setter private Path processedDirPath;
        @Getter @Setter private Path segmentedDirPath;

        public FileGroup(String groupName) {
            this.groupName = groupName;
        }

        public int getFilesCount() {
            return originalFiles.size();
        }

        public String getOriginalFilePath(int index) {
            return originalFiles.get(index);
        }

        public String getProcessedFilePath(int index) {
            return processedFiles.get(index);
        }

        public void addOriginalFile(int index, String filePath) {
            originalFiles.put(index, filePath);
        }

        public void addProcessedFile(int index, String filePath) {
            processedFiles.put(index, filePath);
        }

        public String getSegmentedFilePath(int index) {
            return segmentedFiles.get(index);
        }

        public void addSegmentedFile(int index, String filePath) {
            segmentedFiles.put(index, filePath);
        }
    }
}