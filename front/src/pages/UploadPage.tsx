import { useState } from "react";
import { useNavigate } from "react-router-dom";

function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const navigate = useNavigate();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      setFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleStart = () => {
    // Ici on pourrait stocker les images dans un contexte ou global state
    navigate("/correction");
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>PlantSAM</h1>
      <p>Select the images to be processed</p>

      <input type="file" multiple onChange={handleFileChange} style={{ display: "none" }} id="fileInput"/>
      <label htmlFor="fileInput" style={{
        background: "#333",
        color: "white",
        padding: "10px 20px",
        borderRadius: "8px",
        cursor: "pointer"
      }}>Browse</label>

      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          margin: "20px auto",
          width: "400px",
          height: "200px",
          border: "2px dashed black",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "gray"
        }}
      >
        Drop the files here
      </div>

      <button 
        onClick={handleStart}
        style={{ padding: "10px 20px", borderRadius: "8px", background: "#333", color: "white" }}
      >
        Start
      </button>
    </div>
  );
}

export default UploadPage;
