import { useCallback, useState } from 'react'
import './HomePage.css'

interface HomePageProps {
  onImagesSelected: (images: File[], groupId: string) => void
}

function HomePage({ onImagesSelected }: HomePageProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [groupId, setGroupId] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleFileSelect = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return

    setIsLoading(true)
    const fileArray = Array.from(files)
    setSelectedFiles(fileArray)

    try {
      console.log('Étape 1: Création du groupe...')
      // Create group
      const groupResponse = await fetch('/api/files/group', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `name=plant-group-${Date.now()}`
      })

      if (!groupResponse.ok) {
        throw new Error('Erreur lors de la création du groupe')
      }

      const groupData = await groupResponse.json()
      const newGroupId = groupData.groupId
      setGroupId(newGroupId)
      console.log('Groupe créé avec ID:', newGroupId)

      // Upload files
      console.log('Étape 2: Upload des fichiers...')
      const formData = new FormData()
      fileArray.forEach(file => {
        formData.append('files', file)
      })

      const uploadResponse = await fetch(`/api/files/group/${newGroupId}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        throw new Error('Erreur lors de l\'upload des fichiers')
      }

      console.log('Upload terminé, prêt pour le traitement')
      
    } catch (error) {
      console.error('Erreur détaillée:', error)
      alert('Erreur lors du traitement des images')
      setGroupId('')
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleStartProcessing = async () => {
    if (!groupId || selectedFiles.length === 0) return

    setIsProcessing(true)
    try {
      // Process image
      console.log('Étape 3: Traitement de l\'image...')
      const processResponse = await fetch(`/api/files/group/${groupId}/0/process`, {
        method: 'POST'
      })

      if (!processResponse.ok) {
        throw new Error('Erreur lors du traitement de l\'image')
      }

      console.log('Traitement terminé, passage à la page de correction avec groupId:', groupId)
      onImagesSelected(selectedFiles, groupId)
      
    } catch (error) {
      console.error('Erreur lors du traitement:', error)
      alert('Erreur lors du traitement des images: ')
    } finally {
      setIsProcessing(false)
    }
  }

  const clearSelection = () => {
    setSelectedFiles([])
    setGroupId('')
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFileSelect(e.dataTransfer.files)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  return (
    <div className="home-page">
      <h1>PlantsAM</h1>
      
      <div className="file-section">
        <p>Select the images to be processed</p>
        
        <div className="browse-button">
          <label htmlFor="file-input" className="browse-label">
            {isLoading ? 'Uploading...' : 'Browse Files'}
          </label>
          <input
            id="file-input"
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => handleFileSelect(e.target.files)}
            style={{ display: 'none' }}
            disabled={isLoading || isProcessing}
          />
        </div>

        <p>or</p>

        <div 
          className={`drop-zone ${isDragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <p>{isLoading ? 'Uploading files...' : 'Drop the files here'}</p>
        </div>
      </div>

      <div className="selected-files">
  {selectedFiles.length > 0 && (
    <div>
      <p>{selectedFiles.length} file(s) selected</p>
      <ul>
        {selectedFiles.slice(0, 10).map((file, index) => (
          <li key={index} title={file.name}>
            {file.name.length > 30 ? `${file.name.substring(0, 30)}...` : file.name}
          </li>
        ))}
      </ul>
      {selectedFiles.length > 10 && (
        <p>... and {selectedFiles.length - 10} more files</p>
      )}
      <button 
        className="clear-selection-button" 
        onClick={clearSelection}
        disabled={isProcessing}
      >
        Clear Selection
      </button>
    </div>
  )}
  {isLoading && (
    <p>Uploading images... Please wait</p>
  )}
</div>


      <div className="start-section">
        <button 
          className="start-button" 
          onClick={handleStartProcessing}
          disabled={selectedFiles.length === 0 || !groupId || isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Start Processing'}
        </button>
      </div>
    </div>
  )
}

export default HomePage
