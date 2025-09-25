package fr.ird.plantsam.controller;
import java.io.IOException;

import fr.ird.plantsam.service.FileService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/files/")
public class FileController {
    
    private final FileService fileService;

    public FileController(FileService fileService) {
        this.fileService = fileService;
    }

    @PostMapping("/group")
    public ResponseEntity<String> addGroup(@RequestParam("name") String name) {
        String groupId = fileService.addGroup(name);
        return ResponseEntity.ok("{\"status\": \"success\", \"groupId\": \"" + groupId + "\"}");
    }


    @PostMapping("/group/{groupId}/upload")
    public ResponseEntity<String> fileUpload(@PathVariable("groupId") String groupId,
                                             @RequestParam("files") MultipartFile[] files) {
        try {
            for (MultipartFile file : files) {
                fileService.addFile(groupId, file);
            }
            return ResponseEntity.ok("{\"status\": \"success\"}");
        } catch (IOException e) {
            return ResponseEntity.badRequest().body("Uploading error: " + e.getMessage());
        }
    }

    @PostMapping("/group/{groupId}/{fileId}/process")
    public ResponseEntity<String> processCurrentImage(@PathVariable("groupId") String groupId,
                                                      @PathVariable("fileId") int fileId) throws IOException {
        try {
            if (fileService.processImage(groupId, fileId)) {
                return ResponseEntity.ok("{\"status\": \"success\", " +
                        "\"message\": \"file successfully processed\"}");
            }
            return ResponseEntity.badRequest().body("Processing error: group or file does not exist");

        } catch (IOException e) {
            return ResponseEntity.badRequest().body("Image processing error: " + e.getMessage());
        }
    }

    @PostMapping("/group/{groupId}/{fileId}/point/positive")
    public ResponseEntity<String> setPositivePoint(@PathVariable("groupId") String groupId,
                                                   @PathVariable("fileId") int fileId,
                                                   @RequestParam("x") int x,
                                                   @RequestParam("y") int y) throws IOException {
        if (fileService.setPoint(groupId, fileId, x, y, true)) {
            return ResponseEntity.ok("{\"status\": \"success\", \"message\": \"positive point applied\"}");
        }
        return ResponseEntity.badRequest().body("Error applying positive point");
    }

    @PostMapping("/group/{groupId}/{fileId}/point/negative")
    public ResponseEntity<String> setNegativePoint(@PathVariable("groupId") String groupId,
                                                   @PathVariable("fileId") int fileId,
                                                   @RequestParam("x") int x,
                                                   @RequestParam("y") int y) throws IOException {
        if (fileService.setPoint(groupId, fileId, x, y, false)) {
            return ResponseEntity.ok("{\"status\": \"success\", \"message\": \"negative point applied\"}");
        }
        return ResponseEntity.badRequest().body("Error applying negative point");
    }

    @GetMapping("/group/{groupId}/{fileId}/result")
    public ResponseEntity<byte[]> getProcessedImage(@PathVariable("groupId") String groupId,
                                                    @PathVariable("fileId") int fileId) {
        try {
            byte[] fileData = fileService.getProcessedImage(groupId, fileId);
            if (fileData == null) {
                return ResponseEntity.notFound().build();
            }

            return ResponseEntity.ok()
                    .header("Content-Type", "application/octet-stream")
                    .header("Content-Disposition",
                            "attachment; filename=\"processed-file-" + fileId + "\"")
                    .body(fileData);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(null);
        }
    }

    @DeleteMapping("/group/{groupId}/delete")
    public ResponseEntity<String> deleteGroup(@PathVariable("groupId") String groupId) {
        try {
            if (fileService.deleteGroup(groupId)) {
                return ResponseEntity.ok("{\"status\": \"success\", \"message\": \"Group successfully deleted\"}");
            } else {
                return ResponseEntity.badRequest().body("Group deletion error: " + groupId);
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Group deletion error: " + e.getMessage());
        }
    }

    @DeleteMapping("/group/{groupId}/{fileId}/delete")
    public ResponseEntity<String> deleteFile(@PathVariable("groupId") String groupId,
                                             @PathVariable("fileId")  int fileId) {
        try {
            if (fileService.deleteFile(groupId, fileId)) {
                return ResponseEntity.ok("{\"status\": \"success\", \"message\": \"File successfully deleted\"}");
            } else {
                return ResponseEntity.badRequest().body("File deletion error: " + fileId);
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("File deletion error: " + e.getMessage());
        }
    }

}
