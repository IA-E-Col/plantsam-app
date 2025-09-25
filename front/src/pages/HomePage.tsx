import { useCallback, useState } from 'react'
import './HomePage.css'

interface HomePageProps {
  onImagesSelected: (images: File[]) => void
  onProceed: () => void
}

function HomePage({ onImagesSelected, onProceed }: HomePageProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (files) {
      const fileArray = Array.from(files)
      setSelectedFiles(fileArray)
      onImagesSelected(fileArray)
    }
  }, [onImagesSelected])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    handleFileSelect(e.dataTransfer.files)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }, [])

  return (
    <div className="home-page">
      <h1>PlantsAM</h1>
      
      <div className="file-section">
        <p>Select the images to be processed</p>
        
        <div className="browse-button">
          <label htmlFor="file-input" className="browse-label">
            Browse
          </label>
          <input
            id="file-input"
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => handleFileSelect(e.target.files)}
            style={{ display: 'none' }}
          />
        </div>

        <div 
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <p>Drop the files here</p>
        </div>
      </div>

      <div className="selected-files">
        {selectedFiles.length > 0 && (
          <p>{selectedFiles.length} file(s) selected</p>
        )}
      </div>

      <div className="start-section">
        <button 
          className="start-button" 
          onClick={onProceed}
          disabled={selectedFiles.length === 0}
        >
          Start
        </button>
      </div>
    </div>
  )
}

export default HomePage