import { useCallback, useState, useEffect } from 'react'
import './HomePage.css'

interface Project {
  id: string
  name: string
  imageCount: number
  createdAt: Date
}

interface HomePageProps {
  onImagesSelected: (images: File[], groupId: string) => void
}

function HomePage({ onImagesSelected }: HomePageProps) {
  const [projects, setProjects] = useState<Project[]>([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null)
  const [newProjectName, setNewProjectName] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  // Load projects from localStorage on mount
  useEffect(() => {
    const savedProjects = localStorage.getItem('plantsamProjects')
    if (savedProjects) {
      try {
        const parsed = JSON.parse(savedProjects)
        // Convert date strings back to Date objects
        const projectsWithDates = parsed.map((p: any) => ({
          ...p,
          createdAt: new Date(p.createdAt)
        }))
        setProjects(projectsWithDates)
      } catch (error) {
        console.error('Error loading projects:', error)
      }
    }
  }, [])

  // Save projects to localStorage whenever they change
  useEffect(() => {
    if (projects.length > 0) {
      localStorage.setItem('plantsamProjects', JSON.stringify(projects))
    }
  }, [projects])

  const handleCreateProject = async () => {
    if (!newProjectName.trim() || selectedFiles.length === 0 || isCreating) return

    setIsCreating(true)
    try {
      // Step 1: Create group on backend
      console.log('Creating group...')
      const groupResponse = await fetch('/api/files/group', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `name=${encodeURIComponent(newProjectName.trim())}`
      })

      if (!groupResponse.ok) {
        throw new Error('Failed to create group')
      }

      const groupData = await groupResponse.json()
      const groupId = groupData.groupId
      console.log('Group created with ID:', groupId)

      // Step 2: Upload files to the group
      console.log('Uploading files...')
      const formData = new FormData()
      selectedFiles.forEach(file => {
        formData.append('files', file)
      })

      const uploadResponse = await fetch(`/api/files/group/${groupId}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload files')
      }

      console.log('Files uploaded successfully')

      // Step 3: Process first image
      console.log('Processing first image...')
      const processResponse = await fetch(`/api/files/group/${groupId}/0/process`, {
        method: 'POST'
      })

      if (!processResponse.ok) {
        console.warn('Initial processing failed, but project created')
      }

      // Step 4: Add project to local state
      const newProject: Project = {
        id: groupId,
        name: newProjectName.trim(),
        imageCount: selectedFiles.length,
        createdAt: new Date()
      }
      
      // Save to localStorage immediately before navigation
      const updatedProjects = [newProject, ...projects]
      setProjects(updatedProjects)
      localStorage.setItem('plantsamProjects', JSON.stringify(updatedProjects))
      console.log('Project saved to localStorage:', newProject)
      
      // Close modal and reset form
      setShowCreateModal(false)
      setNewProjectName('')
      const filesToOpen = [...selectedFiles]
      setSelectedFiles([])
      
      console.log('Project created successfully')
      
      // Navigate directly to CorrectionPage with the files
      onImagesSelected(filesToOpen, groupId)
    } catch (error) {
      console.error('Error creating project:', error)
      alert(`Error creating project: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsCreating(false)
    }
  }

  const handleDeleteProject = (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setProjectToDelete(projectId)
    setShowDeleteConfirm(true)
  }

  const confirmDelete = async () => {
    if (!projectToDelete) return

    try {
      // Delete from backend - this will remove all folders (uploads, processed, segmented)
      const response = await fetch(`/api/files/group/${projectToDelete}/delete`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        throw new Error('Failed to delete project from backend')
      }

      console.log('Project deleted from backend successfully')
      
      // Remove from local storage
      setProjects(prev => prev.filter(p => p.id !== projectToDelete))
      
      // Update localStorage
      const updatedProjects = projects.filter(p => p.id !== projectToDelete)
      if (updatedProjects.length > 0) {
        localStorage.setItem('plantsamProjects', JSON.stringify(updatedProjects))
      } else {
        localStorage.removeItem('plantsamProjects')
      }
      
      setShowDeleteConfirm(false)
      setProjectToDelete(null)
    } catch (error) {
      console.error('Error deleting project:', error)
      alert('Error deleting project: ' + (error instanceof Error ? error.message : String(error)))
    }
  }

  const cancelDelete = () => {
    setShowDeleteConfirm(false)
    setProjectToDelete(null)
  }

  const handleOpenProject = async (projectId: string) => {
    const project = projects.find(p => p.id === projectId)
    if (!project) return

    setIsLoading(true)
    try {
      // First, get the actual file count from the backend
      const countResponse = await fetch(`/api/files/group/${projectId}/count`)
      if (!countResponse.ok) {
        throw new Error('Failed to get file count from backend')
      }
      
      const countData = await countResponse.json()
      const actualImageCount = countData.count
      
      console.log(`Loading project ${project.name}: ${actualImageCount} images`)
      
      // Fetch all original images for this project from the backend
      const files: File[] = []
      
      for (let i = 0; i < actualImageCount; i++) {
        try {
          const response = await fetch(`/api/files/group/${projectId}/${i}/original`)
          if (response.ok) {
            const blob = await response.blob()
            // Try to determine the file extension from the content-type
            const contentType = response.headers.get('content-type') || 'image/png'
            const extension = contentType.split('/')[1] || 'png'
            const file = new File([blob], `image_${i}.${extension}`, { type: contentType })
            files.push(file)
          } else {
            console.error(`Failed to load image ${i}:`, response.status)
          }
        } catch (error) {
          console.error(`Error loading image ${i}:`, error)
        }
      }

      if (files.length === 0) {
        throw new Error('No images found for this project')
      }

      // Update the cached imageCount in localStorage if it changed
      if (project.imageCount !== actualImageCount) {
        const updatedProjects = projects.map(p => 
          p.id === projectId ? { ...p, imageCount: actualImageCount } : p
        )
        setProjects(updatedProjects)
        localStorage.setItem('plantsamProjects', JSON.stringify(updatedProjects))
        console.log(`Updated imageCount for project ${project.name}: ${project.imageCount} -> ${actualImageCount}`)
      }

      // Navigate to CorrectionPage with the loaded files
      onImagesSelected(files, projectId)
    } catch (error) {
      console.error('Error opening project:', error)
      alert(`Error opening project: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return
    const fileArray = Array.from(files)
    setSelectedFiles(fileArray)
  }, [])

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
      <header className="home-header">
        <h1>PlantSAM</h1>
        <button 
          className="create-project-button"
          onClick={() => setShowCreateModal(true)}
        >
          + New Project
        </button>
      </header>

      <div className="projects-section">
        <h2>My Projects</h2>
        <div className="projects-grid">
          {projects.map(project => (
            <div 
              key={project.id}
              className="project-card"
              onClick={() => handleOpenProject(project.id)}
            >
              <button
                className="delete-project-button"
                onClick={(e) => handleDeleteProject(project.id, e)}
                title="Delete project"
              >
              </button>
              <div className="project-icon">📁</div>
              <h3 className="project-name">{project.name}</h3>
              <div className="project-info">
                <span className="image-count">{project.imageCount} images</span>
                <span className="project-date">
                  {project.createdAt.toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Create New Project</h2>
            
            <div className="modal-form">
              <div className="form-group">
                <label htmlFor="project-name">Project Name</label>
                <input
                  id="project-name"
                  type="text"
                  placeholder="Enter project name..."
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  className="project-name-input"
                />
              </div>

              <div className="form-group">
                <label>Select Images</label>
                <div className="file-upload-section">
                  <label htmlFor="file-input" className="browse-label">
                    Browse Files
                  </label>
                  <input
                    id="file-input"
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={(e) => handleFileSelect(e.target.files)}
                    style={{ display: 'none' }}
                  />

                  <div 
                    className={`drop-zone ${isDragOver ? 'drag-over' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                  >
                    <p>or drop files here</p>
                  </div>
                </div>

                {selectedFiles.length > 0 && (
                  <div className="selected-files-preview">
                    <p>{selectedFiles.length} file(s) selected</p>
                    <ul>
                      {selectedFiles.slice(0, 5).map((file, index) => (
                        <li key={index}>
                          {file.name.length > 40 ? `${file.name.substring(0, 40)}...` : file.name}
                        </li>
                      ))}
                    </ul>
                    {selectedFiles.length > 5 && (
                      <p className="more-files">... and {selectedFiles.length - 5} more</p>
                    )}
                  </div>
                )}
              </div>

              <div className="modal-actions">
                <button 
                  className="cancel-button"
                  onClick={() => {
                    setShowCreateModal(false)
                    setNewProjectName('')
                    setSelectedFiles([])
                  }}
                >
                  Cancel
                </button>
                <button 
                  className="create-button"
                  onClick={handleCreateProject}
                  disabled={!newProjectName.trim() || selectedFiles.length === 0 || isCreating}
                >
                  {isCreating ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div className="modal-overlay" onClick={cancelDelete}>
          <div className="delete-confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <h3>Delete Project</h3>
            <p>Are you sure you want to delete this project? This action cannot be undone.</p>
            <div className="modal-actions">
              <button className="cancel-button" onClick={cancelDelete}>
                Cancel
              </button>
              <button className="delete-confirm-button" onClick={confirmDelete}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="modal-overlay">
          <div className="loading-dialog">
            <div className="loading-spinner"></div>
            <p>Loading project...</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage