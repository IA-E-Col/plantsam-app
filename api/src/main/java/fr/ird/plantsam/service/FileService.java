package fr.ird.plantsam.service;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.io.FileUtils;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.RestClientException;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
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
    private Path projectsBasePath = Paths.get("projects");
    private RestTemplate restTemplate;

    public FileService(
            @Value("${file.projects-dir:projects}") String projectsDir) {
        this.projectsBasePath = Paths.get(projectsDir).toAbsolutePath().normalize();
        fileGroups = new HashMap<>();
        
        // Configure RestTemplate with timeouts
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10000);
        factory.setReadTimeout(120000);
        this.restTemplate = new RestTemplate(factory);

        try {
            Files.createDirectories(projectsBasePath);
            
            // Reconstruct fileGroups from existing disk folders
            reconstructFileGroupsFromDisk();
        } catch (IOException e) {
            throw new RuntimeException("Could not create projects directory", e);
        }
    }
    
    /**
     * Reconstructs the fileGroups HashMap from existing folders on disk.
     * This is called on service initialization to restore projects after server restart.
     * New structure: projects/{projectName}/originals/ and projects/{projectName}/segmented/
     */
    private void reconstructFileGroupsFromDisk() throws IOException {
        System.out.println("🔄 Reconstructing file groups from disk...");
        
        // Scan projects directory for project folders
        if (!Files.exists(projectsBasePath)) {
            System.out.println("⚠️ No projects directory found, skipping reconstruction");
            return;
        }
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(projectsBasePath)) {
            for (Path projectDir : stream) {
                if (!Files.isDirectory(projectDir)) continue;
                
                String projectName = projectDir.getFileName().toString();
                
                // Load metadata to get groupId
                String groupId = loadGroupIdFromMetadata(projectDir);
                if (groupId == null) {
                    System.out.println("⚠️ Skipping project without metadata: " + projectName);
                    continue;
                }
                
                // Get originals and segmented directories
                Path originalsDir = projectDir.resolve("originals");
                Path segmentedDir = projectDir.resolve("segmented");
                
                if (!Files.exists(originalsDir)) {
                    System.out.println("⚠️ Skipping project without originals folder: " + projectName);
                    continue;
                }
                
                // Get files from the originals directory
                List<Path> originalFiles = new ArrayList<>();
                try (DirectoryStream<Path> fileStream = Files.newDirectoryStream(originalsDir)) {
                    for (Path file : fileStream) {
                        if (Files.isRegularFile(file)) {
                            // Skip metadata files
                            String fileName = file.getFileName().toString();
                            if (!fileName.startsWith(".")) {
                                originalFiles.add(file);
                            }
                        }
                    }
                }
                
                if (originalFiles.isEmpty()) {
                    System.out.println("⚠️ Skipping empty project: " + projectName);
                    continue;
                }
                
                // Sort files by their index suffix (e.g., image_0.png, image_1.png)
                originalFiles.sort((a, b) -> {
                    String nameA = a.getFileName().toString();
                    String nameB = b.getFileName().toString();
                    
                    // Extract index from filename (format: basename_index.ext)
                    int indexA = extractFileIndex(nameA);
                    int indexB = extractFileIndex(nameB);
                    
                    return Integer.compare(indexA, indexB);
                });
                
                // Create FileGroup
                FileGroup fileGroup = new FileGroup(projectName);
                fileGroup.setProjectDirPath(projectDir);
                fileGroup.setOriginalsDirPath(originalsDir);
                fileGroup.setSegmentedDirPath(segmentedDir);
                
                // Load processed indices
                Set<Integer> processedIndices = loadProcessedIndices(projectDir);
                for (Integer idx : processedIndices) {
                    fileGroup.markAsProcessed(idx);
                }
                
                // Add all files to the group
                for (int i = 0; i < originalFiles.size(); i++) {
                    Path file = originalFiles.get(i);
                    fileGroup.addOriginalFile(i, file.toString());
                    
                    // Check if segmented file exists
                    String originalFileName = file.getFileName().toString();
                    String baseName = FilenameUtils.removeExtension(originalFileName);
                    String segmentedFileName = baseName + "_mask.png";
                    Path segmentedFile = segmentedDir.resolve(segmentedFileName);
                    if (Files.exists(segmentedFile)) {
                        fileGroup.addSegmentedFile(i, segmentedFile.toString());
                    }
                }
                
                fileGroups.put(groupId, fileGroup);
                System.out.println("✅ Reconstructed project: " + projectName + " (" + groupId + ") with " + originalFiles.size() + " files");
            }
        }
        
        System.out.println("✅ Reconstruction complete. Total projects: " + fileGroups.size());
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
    


    public String addGroup(String groupName) {
        String groupId = UUID.randomUUID().toString();
        FileGroup newGroup = new FileGroup(groupName);
        try {
            // Create project directory: projects/{projectName}/
            Path projectDir = projectsBasePath.resolve(groupName);
            Path originalsDir = projectDir.resolve("originals");
            Path segmentedDir = projectDir.resolve("segmented");
            
            Files.createDirectories(originalsDir);
            Files.createDirectories(segmentedDir);

            newGroup.setProjectDirPath(projectDir);
            newGroup.setOriginalsDirPath(originalsDir);
            newGroup.setSegmentedDirPath(segmentedDir);
            
            // Save project metadata to disk for persistence across restarts
            saveProjectMetadata(projectDir, groupId, groupName);
        } catch (Exception e) {
            throw new RuntimeException("Could not create project directories", e);
        }
        fileGroups.put(groupId, newGroup);
        return groupId;
    }
    
    /**
     * Saves project metadata (groupId and projectName) to disk
     */
    private void saveProjectMetadata(Path projectDir, String groupId, String projectName) throws IOException {
        Path metadataFile = projectDir.resolve(".metadata");
        String content = groupId + "\n" + projectName;
        Files.writeString(metadataFile, content, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
    
    /**
     * Saves processed indices to disk for persistence
     */
    private void saveProcessedIndices(Path projectDir, Set<Integer> processedIndices) throws IOException {
        Path processedFile = projectDir.resolve(".processed");
        String content = processedIndices.stream()
                .map(String::valueOf)
                .reduce((a, b) -> a + "," + b)
                .orElse("");
        Files.writeString(processedFile, content, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
    
    /**
     * Gets the path to the info file for a specific image, creating the infos directory if needed
     */
    private Path getInfoFilePath(String groupId, int fileIndex) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null || group.getProjectDirPath() == null) {
            throw new IOException("Group not found or project directory not set");
        }
        
        // Create infos directory if it doesn't exist
        Path infosDir = group.getProjectDirPath().resolve("infos");
        if (!Files.exists(infosDir)) {
            Files.createDirectories(infosDir);
        }
        
        // Get the original file name to create corresponding info file name
        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) {
            throw new IOException("Original file not found for index: " + fileIndex);
        }
        
        String fileName = Paths.get(originalFilePath).getFileName().toString();
        String baseName = FilenameUtils.removeExtension(fileName);
        String infoFileName = baseName + ".txt";
        
        return infosDir.resolve(infoFileName);
    }
    
    /**
     * Logs a correction action to the info file for a specific image
     */
    private void logCorrection(String groupId, int fileIndex, String correctionLine) {
        try {
            Path infoFile = getInfoFilePath(groupId, fileIndex);
            
            // Append the correction line to the file
            Files.writeString(infoFile, correctionLine + "\n", 
                StandardOpenOption.CREATE, 
                StandardOpenOption.APPEND);
            
            System.out.println("✅ Logged correction to: " + infoFile);
        } catch (IOException e) {
            System.err.println("❌ Failed to log correction: " + e.getMessage());
        }
    }
    
    /**
     * Loads processed indices from disk
     */
    private Set<Integer> loadProcessedIndices(Path projectDir) {
        try {
            Path processedFile = projectDir.resolve(".processed");
            if (Files.exists(processedFile)) {
                String content = Files.readString(processedFile).trim();
                if (!content.isEmpty()) {
                    Set<Integer> indices = new HashSet<>();
                    for (String index : content.split(",")) {
                        try {
                            indices.add(Integer.parseInt(index.trim()));
                        } catch (NumberFormatException e) {
                            // Skip invalid entries
                        }
                    }
                    return indices;
                }
            }
        } catch (IOException e) {
            System.err.println("⚠️ Error reading processed indices for project " + projectDir.getFileName() + ": " + e.getMessage());
        }
        return new HashSet<>();
    }
    
    /**
     * Loads groupId from metadata file in project directory
     */
    private String loadGroupIdFromMetadata(Path projectDir) {
        try {
            Path metadataFile = projectDir.resolve(".metadata");
            if (Files.exists(metadataFile)) {
                String content = Files.readString(metadataFile).trim();
                // First line is groupId, second line is project name
                String[] lines = content.split("\n");
                if (lines.length > 0) {
                    return lines[0].trim();
                }
            }
        } catch (IOException e) {
            System.err.println("⚠️ Error reading metadata for project " + projectDir.getFileName() + ": " + e.getMessage());
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
        Path destinationFile = group.getOriginalsDirPath().resolve(storedFileName);

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

    public Set<Integer> getProcessedIndices(String groupId) {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            return new HashSet<>();
        }
        return group.getProcessedIndices();
    }

    private void markImageAsProcessed(String groupId, int fileIndex) {
        FileGroup group = fileGroups.get(groupId);
        if (group != null) {
            group.markAsProcessed(fileIndex);
            try {
                saveProcessedIndices(group.getProjectDirPath(), group.getProcessedIndices());
            } catch (IOException e) {
                System.err.println("⚠️ Error saving processed indices: " + e.getMessage());
            }
        }
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
            // Save final mask to segmented folder (no intermediate storage)
            saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
            
            // Mark image as processed
            markImageAsProcessed(groupId, fileIndex);
            
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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Log the point correction
                String pointType = positive ? "positive_point" : "negative_point";
                logCorrection(groupId, fileIndex, pointType + " " + x + "," + y);
                
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

            // Return the final segmented mask
            String filePath = group.getSegmentedFilePath(fileIndex);
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
            
            if (group != null && group.getProjectDirPath() != null) {
                // Delete the entire project directory (contains originals and segmented subfolders)
                FileSystemUtils.deleteRecursively(group.getProjectDirPath());
                System.out.println("✅ Deleted project directory: " + group.getProjectDirPath());
            }
            
            fileGroups.remove(groupId);
            System.out.println("✅ Successfully deleted project: " + groupId);
            return true;
        } catch (IOException e) {
            System.err.println("❌ Error deleting project: " + e.getMessage());
            return false;
        }
    }

    public boolean deleteFile(String groupId, int fileIndex) {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) {
            return false;
        }
        
        // Delete original file
        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath != null) {
            try {
                Files.deleteIfExists(Paths.get(originalFilePath));
            } catch (IOException e) {
                return false;
            }
        }

        // Delete segmented file if it exists
        String segmentedFilePath = group.getSegmentedFilePath(fileIndex);
        if (segmentedFilePath != null) {
            try {
                Files.deleteIfExists(Paths.get(segmentedFilePath));
            } catch (IOException e) {
                // Continue even if segmented file deletion fails
            }
        }

        group.originalFiles.remove(fileIndex);
        group.segmentedFiles.remove(fileIndex);
        return true;
    }

    public boolean segmentWithPoints(String groupId, int fileIndex, String positivePoints, String negativePoints, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Segment avec points - Positifs: " + positivePoints + ", Négatifs: " + negativePoints + ", StartType: " + startType);

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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                // Log all points
                if (positivePoints != null && !positivePoints.equals("[]")) {
                    logCorrection(groupId, fileIndex, "positive_points " + positivePoints);
                }
                if (negativePoints != null && !negativePoints.equals("[]")) {
                    logCorrection(groupId, fileIndex, "negative_points " + negativePoints);
                }
                
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
                
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                // Log the union point
                logCorrection(groupId, fileIndex, "union_point " + x + "," + y);
                
                return true;
            } else {
                System.out.println("Segment union: échec, statut HTTP: " + response.getStatusCode());
                return false;
            }
        } catch (RestClientException e) {
            System.err.println("❌ Erreur lors de l'appel à l'API Python (union): " + e.getMessage());
            if (e.getMessage().contains("Unexpected end of file")) {
                System.err.println("⚠️ Le serveur Python s'est arrêté de manière inattendue. Vérifiez les logs Python.");
            }
        } catch (Exception e) {
            System.err.println("❌ Erreur inattendue lors de segment_union: " + e.getMessage());
            e.printStackTrace();
        }
        return false;
    }

    public boolean segmentIntersection(String groupId, int fileIndex, int x, int y, int pointCount, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Segment intersection - Point: (" + x + ", " + y + "), Count: " + pointCount + ", StartType: " + startType);

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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                // Log the intersection point
                logCorrection(groupId, fileIndex, "intersection_point " + x + "," + y);
                
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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Erreur lors de l'appel à l'API Python: " + e.getMessage());
        }
        return false;
    }



    public boolean removeRectangle(String groupId, int fileIndex, int x, int y, int width, int height, String startType) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Remove rectangle - Coords: (" + x + ", " + y + "), Size: " + width + "x" + height + ", StartType: " + startType);

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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                // Log the rectangle removal with all four corners
                int x1 = x;
                int y1 = y;
                int x2 = x + width;
                int y2 = y;
                int x3 = x + width;
                int y3 = y + height;
                int x4 = x;
                int y4 = y + height;
                logCorrection(groupId, fileIndex, 
                    "rectangle " + x1 + "," + y1 + " " + x2 + "," + y2 + " " + 
                    x3 + "," + y3 + " " + x4 + "," + y4);
                
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
                // Note: clearRectangles typically reverts to initial state, 
                // so we may not want to save a new mask here
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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
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
                // Save final mask to segmented folder (no intermediate storage)
                saveFinalMask(group, fileIndex, response.getBody(), originalFilePath);
                
                // Mark image as processed
                markImageAsProcessed(groupId, fileIndex);
                
                return true;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API for IoU: " + e.getMessage());
            // Clean up the temporary file in case of error
            Files.deleteIfExists(tempPreviousMask);
        }
        return false;
    }

    public boolean restoreMask(String groupId, int fileIndex, byte[] maskData) throws IOException {
        FileGroup group = fileGroups.get(groupId);
        if (group == null) return false;

        String originalFilePath = group.getOriginalFilePath(fileIndex);
        if (originalFilePath == null) return false;

        System.out.println("Restoring previous mask for undo operation");

        try {
            // Save the provided mask data directly to the segmented folder
            saveFinalMask(group, fileIndex, maskData, originalFilePath);
            
            // Mark image as processed
            markImageAsProcessed(groupId, fileIndex);
            
            System.out.println("✅ Mask restored successfully");
            return true;
        } catch (Exception e) {
            System.err.println("Error restoring mask: " + e.getMessage());
            return false;
        }
    }

    private static class FileGroup {
        @Getter String groupName;
        private final Map<Integer, String> originalFiles = new HashMap<>();
        private final Map<Integer, String> segmentedFiles = new HashMap<>();
        private final Set<Integer> processedIndices = new HashSet<>();
        @Getter @Setter private Path projectDirPath;
        @Getter @Setter private Path originalsDirPath;
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

        public void addOriginalFile(int index, String filePath) {
            originalFiles.put(index, filePath);
        }

        public String getSegmentedFilePath(int index) {
            return segmentedFiles.get(index);
        }

        public void addSegmentedFile(int index, String filePath) {
            segmentedFiles.put(index, filePath);
        }

        public void markAsProcessed(int index) {
            processedIndices.add(index);
        }

        public boolean isProcessed(int index) {
            return processedIndices.contains(index);
        }

        public Set<Integer> getProcessedIndices() {
            return new HashSet<>(processedIndices);
        }
    }
}