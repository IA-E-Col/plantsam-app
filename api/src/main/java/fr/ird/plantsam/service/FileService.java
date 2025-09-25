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
                "http://localhost:8000/process", // API Python
                HttpMethod.POST,
                requestEntity,
                byte[].class
        );

        if (response.getStatusCode() == HttpStatus.OK) {
            // Sauvegarder l’image résultante
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

        RestTemplate restTemplate = new RestTemplate();

        // Requête multipart avec file + coordonnées
        FileSystemResource fileResource = new FileSystemResource(new File(originalFilePath));
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        body.add("x", x);
        body.add("y", y);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        String url = positive
                ? "http://localhost:8000/positive_point"
                : "http://localhost:8000/negative_point";

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

    // Class to manage a group of files
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
