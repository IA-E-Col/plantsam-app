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

    public FileService(
            @Value("${file.upload-dir:uploads}") String uploadDir,
            @Value("${file.processed-dir:processed}") String processedDir) {
        this.storagePath = Paths.get(uploadDir).toAbsolutePath().normalize();
        this.processedPath = Paths.get(processedDir).toAbsolutePath().normalize();
        fileGroups = new HashMap<>();

        try {
            Files.createDirectories(storagePath);
            Files.createDirectories(processedPath);
        } catch (IOException e) {
            throw new RuntimeException("Could not create upload directories", e);
        }
    }

    public String addGroup(String groupName) {
        String groupId = UUID.randomUUID().toString();
        FileGroup newGroup = new FileGroup(groupName);
        try {
            Path groupUploadDir = storagePath.resolve(groupId);
            Path groupProcessedDir = processedPath.resolve(groupId);
            Files.createDirectories(groupUploadDir);
            Files.createDirectories(groupProcessedDir);

            newGroup.setUploadDirPath(groupUploadDir);
            newGroup.setProcessedDirPath(groupProcessedDir);
        } catch (Exception e) {
            throw new RuntimeException("Could not create upload directories", e);
        }
        fileGroups.put(groupId, newGroup);
        return groupId;
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
            FileSystemUtils.deleteRecursively(storagePath.resolve(groupId));
            FileSystemUtils.deleteRecursively(processedPath.resolve(groupId));
        } catch (IOException e) {
            return false;
        }

        fileGroups.remove(groupId);
        return true;
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
                return true;
            }
        } catch (Exception e) {
            System.err.println("Error calling Python API for union: " + e.getMessage());
            // Clean up the temporary file in case of error
            Files.deleteIfExists(tempPreviousMask);
        }
        return false;
    }

    private static class FileGroup {
        @Getter String groupName;
        private final Map<Integer, String> originalFiles = new HashMap<>();
        private final Map<Integer, String> processedFiles = new HashMap<>();
        @Getter @Setter private Path uploadDirPath;
        @Getter @Setter private Path processedDirPath;

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
    }
}