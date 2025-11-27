import { useState, useRef, useEffect, useCallback } from 'react'
import './CorrectionPage.css'

// Modifier l'interface Point pour inclure les coordonnées relatives
interface Point {
    x: number; // coordonnée d'affichage
    y: number; // coordonnée d'affichage
    type: 'positive' | 'negative';
    id: number;
    relX: number; // coordonnée relative (0-1)
    relY: number; // coordonnée relative (0-1)
}

interface Rectangle {
    x: number;
    y: number;
    width: number;
    height: number;
    relX: number;
    relY: number;
    relWidth: number;
    relHeight: number;
}

interface SegmentationStep {
    id: number;
    imageUrl: string;
    stepName: string;
    timestamp: Date;
}

interface CorrectionPageProps {
    images: File[]
    groupId: string
    onBack: () => void
}

function CorrectionPage({ images, groupId, onBack }: CorrectionPageProps) {
    const [currentImageIndex, setCurrentImageIndex] = useState(0)
    const [imagesList, setImagesList] = useState<File[]>(images)
    // Track backend indices for each image in the list
    // This is needed because when images are deleted from the frontend list,
    // the backend still uses the original sequential indices (0, 1, 2, etc.)
    // backendIndices[i] tells us which backend file index corresponds to frontend position i
    const [backendIndices, setBackendIndices] = useState<number[]>(images.map((_, i) => i))
    const [pointType, setPointType] = useState<'positive' | 'negative'>('positive')
    const [algoType, setAlgoType] = useState<'union' | 'intersection' | 'iou'>('union')
    const [startType, setStartType] = useState<'segmented' | 'scratch'>('segmented')
    const [points, setPoints] = useState<Point[]>([])
    const [rectangles, setRectangles] = useState<Rectangle[]>([]);
    const [processedImageUrl, setProcessedImageUrl] = useState<string>('')
    const [isLoading, setIsLoading] = useState(false)
    const [segmentationSteps, setSegmentationSteps] = useState<SegmentationStep[]>([])
    const [initialSegmentationUrl, setInitialSegmentationUrl] = useState<string>('')
    const [selectedStepId, setSelectedStepId] = useState<number | null>(null)
    const [isProcessingFull, setIsProcessingFull] = useState(false)
    const imageRef = useRef<HTMLImageElement>(null)
    const pointIdCounter = useRef(0)
    const stepIdCounter = useRef(0)
    const [isDrawingRect, setIsDrawingRect] = useState(false);
    const [rectStart, setRectStart] = useState<{ x: number; y: number } | null>(null);
    const [currentRect, setCurrentRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
    const [justDrewRectangle, setJustDrewRectangle] = useState(false);
    const [isImageLoaded, setIsImageLoaded] = useState(false);
    const [imageDisplayInfo, setImageDisplayInfo] = useState<{
        displayedWidth: number;
        displayedHeight: number;
        offsetX: number;
        offsetY: number;
    } | null>(null);

    const [zoomScale, setZoomScale] = useState(1);
    const [zoomPosition, setZoomPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const fileInputRef = useRef<HTMLInputElement>(null);


    const currentImage = imagesList[currentImageIndex]
    // Get the backend file index for the current frontend image
    const currentBackendIndex = backendIndices[currentImageIndex]

    let imageLoadLogCount = 0;

    // Calcul de position avec mémoïsation et prévention des mises à jour inutiles
    const calculateImagePosition = useCallback(() => {
        if (!imageRef.current) return;

        const img = imageRef.current;
        // Utiliser le wrapper non transformé (parent du conteneur de zoom)
        const zoomContainer = img.parentElement;
        const container = zoomContainer?.parentElement;
        if (!container) return;

        const containerRect = container.getBoundingClientRect();
        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;

        const ratio = Math.min(containerWidth / naturalWidth, containerHeight / naturalHeight);
        const displayedWidth = naturalWidth * ratio;
        const displayedHeight = naturalHeight * ratio;

        const offsetX = (containerWidth - displayedWidth) / 2;
        const offsetY = (containerHeight - displayedHeight) / 2;

        setImageDisplayInfo(prev => {
            // Éviter les mises à jour inutiles
            if (prev &&
                prev.displayedWidth === displayedWidth &&
                prev.displayedHeight === displayedHeight &&
                prev.offsetX === offsetX &&
                prev.offsetY === offsetY) {
                return prev;
            }
            return {
                displayedWidth,
                displayedHeight,
                offsetX,
                offsetY
            };
        });
    }, []);

    const handleImageLoad = useCallback(() => {
        if (imageLoadLogCount < 2) {
            console.log('Image chargée, dimensions naturelles:', {
                width: imageRef.current?.naturalWidth,
                height: imageRef.current?.naturalHeight
            });
            imageLoadLogCount++;
        }
        setIsImageLoaded(true);
        calculateImagePosition();
    }, [calculateImagePosition]);

    // Gestion du resize - version corrigée
    useEffect(() => {
        const handleResize = () => {
            if (isImageLoaded) {
                calculateImagePosition();
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [isImageLoaded, calculateImagePosition]);

    // Nettoyage des URLs
    useEffect(() => {
        return () => {
            if (processedImageUrl) URL.revokeObjectURL(processedImageUrl);
            if (initialSegmentationUrl) URL.revokeObjectURL(initialSegmentationUrl);
            segmentationSteps.forEach(step => URL.revokeObjectURL(step.imageUrl));
        };
    }, [processedImageUrl, initialSegmentationUrl, segmentationSteps]);

    // Tente de charger un résultat existant sans réinitialiser le serveur (utile après Full Segmentation depuis Home)
    const tryLoadExistingProcessed = useCallback(async (): Promise<boolean> => {
        if (!groupId) return false;
        try {
            const resp = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`, { cache: 'no-store' });
            if (!resp.ok) return false;
            const blob = await resp.blob();
            if (blob.size === 0) return false;
            const url = URL.createObjectURL(blob);
            setProcessedImageUrl(prev => { if (prev) URL.revokeObjectURL(prev); return url; });
            return true;
        } catch (_) {
            return false;
        }
    }, [groupId, currentBackendIndex]);

    // Effet d'initialisation PRINCIPAL - UN SEUL
    useEffect(() => {
        if (!groupId || images.length === 0) return;

        let isMounted = true;

        const initializePage = async () => {
            // Essayer d'abord de charger un résultat existant sans reset
            const hasExisting = await tryLoadExistingProcessed();

            // Réinitialiser l'état CLIENT
            if (isMounted) {
                setPoints([]);
                setRectangles([]);
                setSelectedStepId(null);
                setIsImageLoaded(false);
            }

            // Nettoyer les URLs précédentes de manière sécurisée
            if (isMounted) {
                if (processedImageUrl) {
                    URL.revokeObjectURL(processedImageUrl);
                    setProcessedImageUrl('');
                }
                if (initialSegmentationUrl) {
                    URL.revokeObjectURL(initialSegmentationUrl);
                    setInitialSegmentationUrl('');
                }
            }

            try {
                // Initialize empty steps list
                if (isMounted) {
                    setSegmentationSteps([]);
                }

                // Charger la segmentation initiale
                await loadInitialSegmentation();

                if (!hasExisting) {
                    // Si aucun résultat existant, reset côté serveur puis charger
                    await resetServerState();
                    await loadProcessedImage();
                    await pollProcessedImage(60, 1000);
                } else {
                    // On a déjà un résultat: éventuellement compléter par polling pour éviter cache
                    await pollProcessedImage(10, 500);
                }

            } catch (error) {
                console.error('Erreur lors de l\'initialisation:', error);
            }
        };

        initializePage();

        return () => {
            isMounted = false;
        };
    }, [groupId, currentBackendIndex]);

    // Debug: surveiller les changements de l'URL traitée
    useEffect(() => {
        console.log('🔍 processedImageUrl a changé:', {
            hasUrl: !!processedImageUrl,
            urlLength: processedImageUrl?.length,
            isLoading
        });
    }, [processedImageUrl, isLoading]);

    // Réinitialiser le zoom quand l'image change
    useEffect(() => {
        setZoomScale(1);
        setZoomPosition({ x: 0, y: 0 });
    }, [currentImageIndex, isImageLoaded]);

    const loadProcessedImage = async () => {
        if (!groupId) return;

        setIsLoading(true);
        try {
            const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`, { cache: 'no-store' });

            if (response.ok) {
                const blob = await response.blob();
                if (blob.size > 0) {
                    const newUrl = URL.createObjectURL(blob);
                    setProcessedImageUrl(prev => {
                        if (prev) URL.revokeObjectURL(prev);
                        return newUrl;
                    });
                } else {
                    // Blob vide: ne pas écraser par un fallback; on laissera le polling récupérer la bonne image
                }
            } else {
                // Réponse non OK: ne pas écraser l'état; le polling tentera à nouveau
            }
        } catch (error) {
            console.error('Erreur lors du chargement de l\'image traitée:', error);
            // Ne pas forcer de fallback ici
        } finally {
            setIsLoading(false);
        }
    };

    const loadInitialSegmentation = async () => {
        if (!groupId) return;

        try {
            const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`);
            if (response.ok) {
                const blob = await response.blob();
                setInitialSegmentationUrl(prev => {
                    if (prev) URL.revokeObjectURL(prev);
                    return URL.createObjectURL(blob);
                });
            }
        } catch (error) {
            console.error('Erreur lors du chargement de la segmentation initiale:', error);
        }
    };

    // Poll the backend for a processed image (when full segmentation was triggered elsewhere)
    const pollProcessedImage = useCallback(async (maxAttempts: number = 20, intervalMs: number = 500) => {
        if (!groupId) return false;
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const resp = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`, { cache: 'no-store' });
                if (resp.ok) {
                    const blob = await resp.blob();
                    if (blob.size > 0) {
                        const url = URL.createObjectURL(blob);
                        setProcessedImageUrl(prev => {
                            if (prev) URL.revokeObjectURL(prev);
                            return url;
                        });
                        return true;
                    }
                }
            } catch (e) {
                // ignore and retry
            }
            await new Promise(r => setTimeout(r, intervalMs));
        }
        return false;
    }, [groupId, currentBackendIndex]);

    const removeSegmentationStep = (id: number) => {
        setSegmentationSteps(prev => {
            const stepToRemove = prev.find(step => step.id === id);
            if (stepToRemove) {
                URL.revokeObjectURL(stepToRemove.imageUrl);
            }
            return prev.filter(step => step.id !== id);
        });

        if (selectedStepId === id) {
            setSelectedStepId(null);
        }
    };

    const saveStepImage = async (imageBlob: Blob, stepName: string) => {
        if (!groupId) return false;

        try {
            const imageBuffer = await imageBlob.arrayBuffer();
            const response = await fetch(
                `/api/files/group/${groupId}/${currentBackendIndex}/save_step?stepName=${encodeURIComponent(stepName)}`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/octet-stream',
                    },
                    body: imageBuffer
                }
            );

            return response.ok;
        } catch (error) {
            console.error('Erreur lors de la sauvegarde de l\'étape:', error);
            return false;
        }
    };

    const applyFullSegmentation = async () => {
        if (!groupId) return;

        setIsProcessingFull(true);
        try {
            const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/process`, { method: 'POST' });

            if (response.ok) {
                const resultResp = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`);
                if (!resultResp.ok) {
                    throw new Error(`Erreur lors du chargement du résultat: ${resultResp.status}`);
                }
                const resultBlob = await resultResp.blob();
                if (resultBlob.size === 0) {
                    throw new Error('Image vide reçue du serveur');
                }

                const newUrl = URL.createObjectURL(resultBlob);
                setProcessedImageUrl(prev => {
                    if (prev) URL.revokeObjectURL(prev);
                    return newUrl;
                });

                const stepName = `full_segmentation_${Date.now()}`;

                // Sauvegarder l'étape avec le blob du résultat
                await saveStepImage(resultBlob, stepName);
                const stepUrl = URL.createObjectURL(resultBlob);
                const newStep: SegmentationStep = {
                    id: stepIdCounter.current++,
                    imageUrl: stepUrl,
                    stepName: stepName,
                    timestamp: new Date()
                };
                setSegmentationSteps(prev => [...prev, newStep]);

                setPoints([]);

                // Forcer un re-rendu
                setCurrentImageIndex(prev => prev);
            } else {
                const errorText = await response.text();
                console.error('Erreur lors de la segmentation totale:', response.status, errorText);
                alert(`Erreur lors de la segmentation totale: ${response.status}`);
            }
        } catch (error) {
            console.error('Erreur API lors de la segmentation totale:', error);
            alert(`Erreur lors de la segmentation totale: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsProcessingFull(false);
        }
    };

    const handleAddImageClick = () => {
        fileInputRef.current?.click();
    };

    const handleDeleteImage = (index: number, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent triggering the image selection
        
        if (imagesList.length === 1) {
            alert('Cannot delete the last image');
            return;
        }

        if (confirm('Are you sure you want to delete this image?')) {
            const isDeletingCurrentImage = index === currentImageIndex;
            
            // Remove from local list
            setImagesList(prev => prev.filter((_, i) => i !== index));
            // Remove corresponding backend index
            setBackendIndices(prev => prev.filter((_, i) => i !== index));
            
            // Reset state if deleting the current image
            if (isDeletingCurrentImage) {
                setPoints([]);
                setRectangles([]);
                setSelectedStepId(null);
                setSegmentationSteps([]);
                if (processedImageUrl) {
                    URL.revokeObjectURL(processedImageUrl);
                    setProcessedImageUrl('');
                }
                if (initialSegmentationUrl) {
                    URL.revokeObjectURL(initialSegmentationUrl);
                    setInitialSegmentationUrl('');
                }
            }
            
            // Adjust current index if needed
            if (index === currentImageIndex) {
                // If deleting current image, move to previous or next
                if (index > 0) {
                    setCurrentImageIndex(index - 1);
                } else {
                    setCurrentImageIndex(0);
                }
            } else if (index < currentImageIndex) {
                // If deleting before current, adjust index
                setCurrentImageIndex(prev => prev - 1);
            }
            
            console.log('Image deleted successfully');
        }
    };

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const newImage = files[0];
        
        // Validate it's an image
        if (!newImage.type.startsWith('image/')) {
            alert('Please select a valid image file');
            return;
        }

        try {
            setIsLoading(true);
            
            // Upload the new image to the backend using the existing upload endpoint
            const formData = new FormData();
            formData.append('files', newImage);
            
            const response = await fetch(`/api/files/group/${groupId}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to upload image');
            }

            // Add to local images list
            setImagesList(prev => [...prev, newImage]);
            
            // Track the backend index for this new image
            // The backend assigns sequential indices based on total uploads
            const newBackendIndex = backendIndices.length > 0 ? Math.max(...backendIndices) + 1 : imagesList.length;
            setBackendIndices(prev => [...prev, newBackendIndex]);
            
            // Reset the page state for the new image
            setPoints([]);
            setRectangles([]);
            setSelectedStepId(null);
            setSegmentationSteps([]);
            if (processedImageUrl) {
                URL.revokeObjectURL(processedImageUrl);
                setProcessedImageUrl('');
            }
            if (initialSegmentationUrl) {
                URL.revokeObjectURL(initialSegmentationUrl);
                setInitialSegmentationUrl('');
            }
            
            // Switch to the new image (it will be at the end of the list)
            const newIndex = imagesList.length;
            setCurrentImageIndex(newIndex);
            
            console.log('New image added successfully with backend index:', newBackendIndex);
        } catch (error) {
            console.error('Error adding image:', error);
            alert(`Error adding image: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsLoading(false);
            // Reset the input
            if (e.target) {
                e.target.value = '';
            }
        }
    };

    const handleImageClick = async (e: React.MouseEvent<HTMLDivElement>) => {
        if (justDrewRectangle || !isImageLoaded || !imageDisplayInfo) {
            return;
        }

        const container = e.currentTarget;
        const rect = container.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        if (clickX < imageDisplayInfo.offsetX ||
            clickX > imageDisplayInfo.offsetX + imageDisplayInfo.displayedWidth ||
            clickY < imageDisplayInfo.offsetY ||
            clickY > imageDisplayInfo.offsetY + imageDisplayInfo.displayedHeight) {
            return;
        }

        // Calculer les coordonnées relatives
        const relX = (clickX - imageDisplayInfo.offsetX) / imageDisplayInfo.displayedWidth;
        const relY = (clickY - imageDisplayInfo.offsetY) / imageDisplayInfo.displayedHeight;

        const newPoint: Point = {
            x: clickX,
            y: clickY,
            relX: relX,
            relY: relY,
            type: pointType,
            id: pointIdCounter.current++
        };

        setPoints(prev => [...prev, newPoint]);

        // Apply only the new point incrementally
        await applyNewPoint(newPoint);
    };

    const getDisplayCoordinates = useCallback((relX: number, relY: number) => {
        if (!imageDisplayInfo) return { x: 0, y: 0 };

        // Toujours retourner les coordonnées de base sans transformation de zoom
        // pour que les points restent fixes visuellement
        return {
            x: relX * imageDisplayInfo.displayedWidth + imageDisplayInfo.offsetX,
            y: relY * imageDisplayInfo.displayedHeight + imageDisplayInfo.offsetY
        };
    }, [imageDisplayInfo]); // Retirer zoomScale et zoomPosition des dépendances

    const getDisplayDimensions = useCallback((relWidth: number, relHeight: number) => {
        if (!imageDisplayInfo) return { width: 0, height: 0 };

        // Toujours retourner les dimensions de base sans transformation de zoom
        return {
            width: relWidth * imageDisplayInfo.displayedWidth,
            height: relHeight * imageDisplayInfo.displayedHeight
        };
    }, [imageDisplayInfo]); // Retirer zoomScale des dépendances

    const getRelativeCoordinates = useCallback((clickX: number, clickY: number, containerCenterX: number, containerCenterY: number) => {
        if (!imageDisplayInfo) return { relX: 0, relY: 0 };

        let adjustedX = clickX;
        let adjustedY = clickY;

        // Inverser la transformation appliquée au conteneur de zoom
        if (zoomScale > 1) {
            // Modèle direct (avec transform: scale(...) translate(...)):
            // screen = center + zoomScale * (local - center) + zoomPosition
            // Inversion:
            // local = center + (screen - center - zoomPosition) / zoomScale
            adjustedX = containerCenterX + (clickX - containerCenterX - zoomPosition.x) / zoomScale;
            adjustedY = containerCenterY + (clickY - containerCenterY - zoomPosition.y) / zoomScale;
        }

        // Calculer les coordonnées relatives par rapport à l'image affichée (avant zoom)
        const relX = (adjustedX - imageDisplayInfo.offsetX) / imageDisplayInfo.displayedWidth;
        const relY = (adjustedY - imageDisplayInfo.offsetY) / imageDisplayInfo.displayedHeight;

        return {
            relX: Math.max(0, Math.min(1, relX)),
            relY: Math.max(0, Math.min(1, relY))
        };
    }, [imageDisplayInfo, zoomScale, zoomPosition]);

    useEffect(() => {
        if (points.length > 0) {
            console.log('Dernier point placé:', points[points.length - 1]);
        }
    }, [points]);

    const handleStepClick = (stepId: number) => {
        setSelectedStepId(stepId);
    };

    const undoLastPoint = async () => {
        if (points.length === 0) return;

        const newPoints = points.slice(0, -1);
        setPoints(newPoints);

        if (newPoints.length === 0) {
            // No points left, just clear
            await clearPoints();
        } else {
            // Rebuild state by clearing and reapplying all remaining points
            setIsLoading(true);
            try {
                // Clear points on backend
                await fetch(
                    `/api/files/group/${groupId}/${currentBackendIndex}/clear_points`,
                    { method: 'POST' }
                );

                // Reapply all remaining points incrementally
                for (const point of newPoints) {
                    await applyNewPoint(point);
                }
            } catch (error) {
                console.error('Error during undo:', error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const clearPoints = async () => {
        try {
            const response = await fetch(
                `/api/files/group/${groupId}/${currentBackendIndex}/clear_points`,
                { method: 'POST' }
            );

            if (response.ok) {
                setPoints([]);
                if (startType === 'segmented') {
                    await loadInitialSegmentation();
                } else {
                    const url = URL.createObjectURL(currentImage);
                    setProcessedImageUrl(url);
                }
            } else {
                console.error('Erreur lors de effacement des points:', response.status);
            }
        } catch (error) {
            console.error('Erreur API:', error);
        }
    };

    const downloadProcessedImage = async () => {
        if (!groupId) return;

        try {
            console.log('=== DÉBUT TÉLÉCHARGEMENT ===');

            // Méthode 1: Télécharger directement depuis l'API
            const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/result`);

            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }

            const blob = await response.blob();
            console.log('Blob reçu:', blob.size, 'bytes, type:', blob.type);

            if (blob.size === 0) {
                throw new Error('Blob vide reçu du serveur');
            }

            // Créer une URL blob pour le téléchargement
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;

            const originalName = images[currentImageIndex].name;
            const nameWithoutExtension = originalName.replace(/\.[^/.]+$/, "");
            a.download = `${nameWithoutExtension}_segmented_${Date.now()}.png`;

            document.body.appendChild(a);
            a.click();

            // Nettoyer après un délai
            setTimeout(() => {
                URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);

            console.log('=== TÉLÉCHARGEMENT RÉUSSI ===');

        } catch (error) {
            console.error('Erreur lors du téléchargement:', error);

            // Méthode de fallback
            try {
                if (processedImageUrl && processedImageUrl.startsWith('blob:')) {
                    const a = document.createElement('a');
                    a.href = processedImageUrl;
                    a.download = `segmented_${Date.now()}.png`;
                    a.click();
                    console.log('Fallback réussi avec URL blob directe');
                }
            } catch (fallbackError) {
                console.error('Fallback échoué:', fallbackError);
                alert('Erreur lors du téléchargement. Veuillez réessayer.');
            }
        }
    };

    const resetServerState = async () => {
        if (!groupId) return;

        try {
            console.log('Réinitialisation de l\'état serveur...');

            // Réinitialiser les points
            await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/clear_points`, {
                method: 'POST'
            });

            // Réinitialiser les rectangles
            await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/clear_rectangles`, {
                method: 'POST'
            });

            console.log('État serveur réinitialisé avec succès');
        } catch (error) {
            console.error('Erreur lors de la réinitialisation serveur:', error);
        }
    };

    const handleManualNoiseRemovalClick = () => {
        if (!isImageLoaded) {
            alert("Veuillez attendre que l'image soit complètement chargée");
            return;
        }
        setIsDrawingRect(true);
        // NE PAS réinitialiser les points : retirer setPoints([]);
    };

    const handleZoomIn = () => {
        setZoomScale(prev => Math.min(prev * 1.5, 10)); // Zoom max 10x
    };

    const handleZoomOut = () => {
        setZoomScale(prev => Math.max(prev / 1.5, 1)); // Zoom min 1x (taille normale)
    };

    const handleResetZoom = () => {
        setZoomScale(1);
        setZoomPosition({ x: 0, y: 0 });
    };

    const handleContextMenu = (e: React.MouseEvent<HTMLDivElement>) => {
        // Empêcher le menu contextuel seulement en mode déplacement zoom
        if (zoomScale > 1) {
            e.preventDefault();
            e.stopPropagation();
        }
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        // Mode dessin de rectangle - seulement avec clic gauche
        if (isDrawingRect && e.button === 0) {
            if (!imageDisplayInfo || !isImageLoaded) return;

            e.preventDefault();
            e.stopPropagation();

            const container = e.currentTarget;
            const rect = container.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            // Utiliser getRelativeCoordinates pour vérifier si le début du rectangle est dans l'image
            const { relX, relY } = getRelativeCoordinates(clickX, clickY, rect.width / 2, rect.height / 2);

            // Vérifier si le début du rectangle est dans les limites de l'image (0-1)
            if (relX < 0 || relX > 1 || relY < 0 || relY > 1) {
                return;
            }

            setRectStart({ x: clickX, y: clickY });
            setCurrentRect({ x: clickX, y: clickY, width: 0, height: 0 });
        }
        // Mode drag pour le zoom - UNIQUEMENT avec clic droit
        else if (e.button === 2 && zoomScale > 1) {
            e.preventDefault();
            e.stopPropagation();
            setIsDragging(true);
            setDragStart({
                x: e.clientX - zoomPosition.x,
                y: e.clientY - zoomPosition.y
            });
        }
        // Dans la fonction handleMouseDown, remplacez cette partie :
        else if (e.button === 0 && !justDrewRectangle && isImageLoaded && imageDisplayInfo && !isDrawingRect) {
            e.preventDefault();
            e.stopPropagation();

            const container = e.currentTarget;
            const rect = container.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            // Utiliser la fonction de conversion qui gère le zoom
            const { relX, relY } = getRelativeCoordinates(clickX, clickY, rect.width / 2, rect.height / 2);

            // Vérifier si le clic est dans les limites de l'image (0-1)
            if (relX < 0 || relX > 1 || relY < 0 || relY > 1) {
                return;
            }

            console.log('Placement de point - coordonnées:', { clickX, clickY, relX, relY, zoomScale, zoomPosition });

            // Obtenir les coordonnées d'affichage de base (sans zoom) pour le point
            const displayCoords = getDisplayCoordinates(relX, relY);

            const newPoint: Point = {
                x: displayCoords.x, // Utiliser les coordonnées d'affichage de base
                y: displayCoords.y, // Utiliser les coordonnées d'affichage de base
                relX: relX,
                relY: relY,
                type: pointType,
                id: pointIdCounter.current++
            };

            setPoints(prev => [...prev, newPoint]);
            // Apply only the new point incrementally
            applyNewPoint(newPoint);
        }
    };

    const getRelativeFromDisplay = useCallback((displayX: number, displayY: number) => {
        if (!imageDisplayInfo) return { relX: 0, relY: 0 };

        const relX = (displayX - imageDisplayInfo.offsetX) / imageDisplayInfo.displayedWidth;
        const relY = (displayY - imageDisplayInfo.offsetY) / imageDisplayInfo.displayedHeight;

        return {
            relX: Math.max(0, Math.min(1, relX)),
            relY: Math.max(0, Math.min(1, relY))
        };
    }, [imageDisplayInfo]);


    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        // Mode dessin de rectangle
        if (isDrawingRect && rectStart && imageDisplayInfo) {
            e.preventDefault();
            e.stopPropagation();

            const container = e.currentTarget;
            const rect = container.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            // Convertir les coordonnées de début et de fin en coordonnées relatives
            const startRel = getRelativeCoordinates(rectStart.x, rectStart.y, rect.width / 2, rect.height / 2);
            const endRel = getRelativeCoordinates(currentX, currentY, rect.width / 2, rect.height / 2);

            // Calculer les dimensions en coordonnées relatives
            const relX = Math.min(startRel.relX, endRel.relX);
            const relY = Math.min(startRel.relY, endRel.relY);
            const relWidth = Math.abs(endRel.relX - startRel.relX);
            const relHeight = Math.abs(endRel.relY - startRel.relY);

            // Obtenir les coordonnées d'affichage pour le rectangle
            const displayCoords = getDisplayCoordinates(relX, relY);
            const displayDims = getDisplayDimensions(relWidth, relHeight);

            setCurrentRect({
                x: displayCoords.x,
                y: displayCoords.y,
                width: displayDims.width,
                height: displayDims.height
            });
        }
        // Mode drag pour le zoom
        else if (isDragging && zoomScale > 1) {
            e.preventDefault();
            const x = e.clientX - dragStart.x;
            const y = e.clientY - dragStart.y;

            if (imageDisplayInfo) {
                const maxX = Math.max(0, (imageDisplayInfo.displayedWidth * zoomScale - imageDisplayInfo.displayedWidth) / 2);
                const maxY = Math.max(0, (imageDisplayInfo.displayedHeight * zoomScale - imageDisplayInfo.displayedHeight) / 2);

                setZoomPosition({
                    x: Math.max(-maxX, Math.min(maxX, x)),
                    y: Math.max(-maxY, Math.min(maxY, y))
                });
            }
        }
    };

    const handleMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
        // Mode dessin de rectangle
        if (isDrawingRect) {
            if (!isImageLoaded || !imageDisplayInfo) return;

            e.preventDefault();
            e.stopPropagation();

            if (currentRect && currentRect.width > 0 && currentRect.height > 0) {
                // Convertir les coordonnées d'affichage du rectangle en coordonnées relatives
                const startRel = getRelativeFromDisplay(currentRect.x, currentRect.y);
                const endRel = getRelativeFromDisplay(
                    currentRect.x + currentRect.width,
                    currentRect.y + currentRect.height
                );

                const relX = startRel.relX;
                const relY = startRel.relY;
                const relWidth = Math.abs(endRel.relX - startRel.relX);
                const relHeight = Math.abs(endRel.relY - startRel.relY);

                const newRectangle: Rectangle = {
                    x: currentRect.x,
                    y: currentRect.y,
                    width: currentRect.width,
                    height: currentRect.height,
                    relX: relX,
                    relY: relY,
                    relWidth: relWidth,
                    relHeight: relHeight
                };

                setRectangles(prev => [...prev, newRectangle]);
                // Apply only the new rectangle incrementally
                applyNewRectangle(newRectangle);
            }

            cancelRectangleMode();
            setJustDrewRectangle(true);
            setTimeout(() => setJustDrewRectangle(false), 100);
        }
        // Mode drag pour le zoom
        else if (isDragging) {
            setIsDragging(false);
        }
    };

    const getZoomStyle = () => {
        if (zoomScale === 1) {
            return {
                position: 'relative' // ← Ajouter même quand pas de zoom
            };
        }
        return {
            transform: `translate(${zoomPosition.x}px, ${zoomPosition.y}px) scale(${zoomScale})`,
            transformOrigin: 'center center',
            cursor: isDragging ? 'grabbing' : 'grab',
            position: 'relative' // ← Ajouter cette ligne
        };
    };


    const cancelRectangleMode = () => {
        setRectStart(null);
        setCurrentRect(null);
        setIsDrawingRect(false);
    };

    // Apply only the new rectangle (incremental approach)
    const applyNewRectangle = async (rect: Rectangle) => {
        if (!groupId) return;

        setIsLoading(true);
        try {
            const img = imageRef.current;
            if (!img) return;

            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;

            const originalRect = {
                x: Math.round(rect.relX * naturalWidth),
                y: Math.round(rect.relY * naturalHeight),
                width: Math.round(rect.relWidth * naturalWidth),
                height: Math.round(rect.relHeight * naturalHeight)
            };

            console.log('Applying new rectangle:', originalRect);

            const response = await fetch(
                `/api/files/group/${groupId}/${currentBackendIndex}/remove_rectangle?x=${originalRect.x}&y=${originalRect.y}&width=${originalRect.width}&height=${originalRect.height}&startType=${startType}`,
                { method: 'POST' }
            );

            if (!response.ok) {
                console.error('Error applying rectangle');
                return;
            }

            const blob = await response.blob();
            
            // Save the step with a meaningful name
            const rectangleIndex = rectangles.length;
            const stepName = `rectangle_${rectangleIndex}_removal`;
            
            await saveStepImage(blob, stepName);
            const stepUrl = URL.createObjectURL(blob);
            const newStep: SegmentationStep = {
                id: stepIdCounter.current++,
                imageUrl: stepUrl,
                stepName: stepName,
                timestamp: new Date()
            };
            setSegmentationSteps(prev => [...prev, newStep]);
            
            if (processedImageUrl) {
                URL.revokeObjectURL(processedImageUrl);
            }
            const url = URL.createObjectURL(blob);
            setProcessedImageUrl(url);

            console.log('Rectangle applied successfully');
        } catch (error) {
            console.error('Error applying rectangle:', error);
            alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsLoading(false);
        }
    };

    // Apply only the new point (incremental approach)
    const applyNewPoint = async (point: Point) => {
        if (!groupId || !imageDisplayInfo) return;

        setIsLoading(true);
        try {
            const img = imageRef.current;
            if (!img) return;

            const naturalWidth = img.naturalWidth;
            const naturalHeight = img.naturalHeight;

            const scaledPoint = {
                x: Math.round(point.relX * naturalWidth),
                y: Math.round(point.relY * naturalHeight)
            };

            console.log('Applying new point:', scaledPoint, 'type:', point.type);

            let response: Response | null = null;

            if (algoType === 'union' && point.type === 'positive') {
                // Get current point count of same type
                const positivePoints = points.filter(p => p.type === 'positive');
                const pointCount = positivePoints.length + 1; // +1 for the new point

                response = await fetch(
                    `/api/files/group/${groupId}/${currentBackendIndex}/segment_union?x=${scaledPoint.x}&y=${scaledPoint.y}&pointCount=${pointCount}&startType=${startType}`,
                    { method: 'POST' }
                );
            } else if (algoType === 'intersection' && point.type === 'negative') {
                const negativePoints = points.filter(p => p.type === 'negative');
                const pointCount = negativePoints.length + 1;

                response = await fetch(
                    `/api/files/group/${groupId}/${currentBackendIndex}/segment_intersection?x=${scaledPoint.x}&y=${scaledPoint.y}&pointCount=${pointCount}&startType=${startType}`,
                    { method: 'POST' }
                );
            } else {
                // For IOU or mismatched types, use segment_with_points with all points
                const allPoints = [...points, point];
                const positivePoints = allPoints.filter(p => p.type === 'positive')
                    .map(p => [Math.round(p.relX * naturalWidth), Math.round(p.relY * naturalHeight)]);
                const negativePoints = allPoints.filter(p => p.type === 'negative')
                    .map(p => [Math.round(p.relX * naturalWidth), Math.round(p.relY * naturalHeight)]);

                const formData = new FormData();
                formData.append('positivePoints', JSON.stringify(positivePoints));
                formData.append('negativePoints', JSON.stringify(negativePoints));
                formData.append('startType', startType);

                response = await fetch(
                    `/api/files/group/${groupId}/${currentBackendIndex}/segment_with_points`,
                    {
                        method: 'POST',
                        body: formData
                    }
                );
            }

            if (response && response.ok) {
                const blob = await response.blob();
                if (blob.size === 0) {
                    throw new Error('Empty image received from server');
                }

                // Save the step with a meaningful name
                const pointType = point.type === 'positive' ? 'positive' : 'negative';
                const pointIndex = points.filter(p => p.type === point.type).length + 1;
                const stepName = `point_${pointIndex}_${algoType}_${pointType}`;
                
                await saveStepImage(blob, stepName);
                const stepUrl = URL.createObjectURL(blob);
                const newStep: SegmentationStep = {
                    id: stepIdCounter.current++,
                    imageUrl: stepUrl,
                    stepName: stepName,
                    timestamp: new Date()
                };
                setSegmentationSteps(prev => [...prev, newStep]);

                if (processedImageUrl) {
                    URL.revokeObjectURL(processedImageUrl);
                }
                const newUrl = URL.createObjectURL(blob);
                setProcessedImageUrl(newUrl);

                console.log('Point applied successfully');
            }
        } catch (error) {
            console.error('Error applying point:', error);
            alert(`Error applying point: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsLoading(false);
        }
    };

    const previousImageUrl = useCallback(() => {
        if (selectedStepId) {
            const step = segmentationSteps.find(step => step.id === selectedStepId);
            return step?.imageUrl || URL.createObjectURL(currentImage);
        }
        if (segmentationSteps.length > 1) {
            return segmentationSteps[segmentationSteps.length - 2]?.imageUrl || URL.createObjectURL(currentImage);
        }
        return segmentationSteps[0]?.imageUrl || URL.createObjectURL(currentImage);
    }, [selectedStepId, segmentationSteps, currentImage]);

    const segmentedImageUrl = useCallback(() => {
        return processedImageUrl ||
            (segmentationSteps.length > 0 ? segmentationSteps[segmentationSteps.length - 1]?.imageUrl : URL.createObjectURL(currentImage));
    }, [processedImageUrl, segmentationSteps, currentImage]);

    const applyAlgorithm = async () => {
        if (!groupId || !currentImage) return;

        setIsLoading(true);
        try {
            console.log('Applying algorithm:', algoType);

            let previousMaskBlob: Blob;
            
            if (selectedStepId) {
                const step = segmentationSteps.find(step => step.id === selectedStepId);
                if (!step) throw new Error("Selected step not found");
                const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/step/${step.stepName}`);
                if (!response.ok) throw new Error("Failed to fetch selected step mask");
                previousMaskBlob = await response.blob();
            } else if (segmentationSteps.length > 0) {
                // Get the last step's mask from the backend
                const lastStep = segmentationSteps[segmentationSteps.length - 1];
                const response = await fetch(`/api/files/group/${groupId}/${currentBackendIndex}/step/${lastStep.stepName}`);
                if (!response.ok) throw new Error("Failed to fetch previous step mask");
                previousMaskBlob = await response.blob();
            } else {
                throw new Error("No previous mask available");
            }

            const formData = new FormData();
            formData.append('previousMask', previousMaskBlob);

            let endpoint: string;
            if (algoType === 'union') {
                endpoint = 'apply_union';
            } else if (algoType === 'intersection') {
                endpoint = 'apply_intersection';
            } else if (algoType === 'iou') {
                endpoint = 'apply_iou';
            } else {
                throw new Error(`Unknown algorithm type: ${algoType}`);
            }

            const response = await fetch(
                `/api/files/group/${groupId}/${currentBackendIndex}/${endpoint}`,
                {
                    method: 'POST',
                    body: previousMaskBlob
                }
            );

            if (!response.ok) {
                throw new Error(`Error applying ${algoType}: ${response.statusText}`);
            }

            // Update the processed image with the result
            const blob = await response.blob();
            if (processedImageUrl) {
                URL.revokeObjectURL(processedImageUrl);
            }
            const url = URL.createObjectURL(blob);
            
            setPoints([]);
            setProcessedImageUrl(url);

            // Save the step and add it to segmentation steps
            const stepName = `${algoType}_operation`;
            await saveStepImage(blob, stepName);
            const stepUrl = URL.createObjectURL(blob);
            const newStep: SegmentationStep = {
                id: stepIdCounter.current++,
                imageUrl: stepUrl,
                stepName: stepName,
                timestamp: new Date()
            };
            setSegmentationSteps(prev => [...prev, newStep]);

            // Reset algorithm type to 'union' so subsequent point placements work correctly
            setAlgoType('union');

            console.log(`${algoType} operation applied successfully`);
        } catch (error) {
            console.error('Error applying algorithm:', error);
            if (error instanceof Error) {
                alert(`Error applying ${algoType}: ${error.message}`);
            } else {
                alert(`Error applying ${algoType}`);
            }
        } finally {
            setIsLoading(false);
        }
    };

    if (!groupId) {
        return (
            <div className="correction-page">
                <div style={{ padding: '2rem', textAlign: 'center' }}>
                    <h2>Erreur: GroupId non défini</h2>
                    <button onClick={onBack}>Retour à l'accueil</button>
                </div>
            </div>
        );
    }

    return (
        <div className="correction-page">
            <header className="correction-header">
                <h1>PlantSAM</h1>
                <div>
                    <button className="back-button" onClick={onBack}>
                        Back
                    </button>
                </div>
            </header>

            <div className="correction-layout">
                <div className="images-sidebar">
                    <h3>Images ({imagesList.length})</h3>
                    <div className="images-list">
                        {imagesList.map((image, index) => (
                            <div
                                key={index}
                                className={`image-item ${index === currentImageIndex ? 'active' : ''}`}
                                onClick={() => setCurrentImageIndex(index)}
                            >
                                <img
                                    src={URL.createObjectURL(image)}
                                    alt={`Thumbnail ${index}`}
                                    className="thumbnail"
                                />
                                {index === currentImageIndex && (
                                    <div className="active-indicator"></div>
                                )}
                                <button
                                    className="delete-image-button"
                                    onClick={(e) => handleDeleteImage(index, e)}
                                    title="Delete image"
                                >
                                </button>
                            </div>
                        ))}
                    </div>
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept="image/*"
                        onChange={handleFileSelect}
                    />
                    <button className="add-image-button" onClick={handleAddImageClick}>
                        + Add Image
                    </button>
                </div>

                <div className="correction-content">
                    <div className="main-content-area">
                        <div className="images-section">
                            <div className="image-container">
                                <h3>Previous mask</h3>
                                <div className="image-wrapper">
                                    <img
                                        src={previousImageUrl() || URL.createObjectURL(currentImage)}
                                        alt="Previous"
                                        className="correction-image"
                                        onError={(e) => {
                                            console.error('Erreur de chargement Previous:', e);
                                            // Fallback vers l'image originale en cas d'erreur
                                            e.currentTarget.src = URL.createObjectURL(currentImage);
                                        }}
                                    />
                                </div>
                            </div>

                            <div className="image-container">
                                <h3>Final Mask {isLoading && '(Loading)'}</h3>
                                <div className="image-wrapper">
                                    <img
                                        src={segmentedImageUrl() || URL.createObjectURL(currentImage)}
                                        alt="Segmented"
                                        className="correction-image"
                                        onError={(e) => {
                                            console.error('Erreur de chargement Segmented:', e);
                                            e.currentTarget.src = URL.createObjectURL(currentImage);
                                        }}
                                    />
                                </div>
                            </div>

                            <div className="image-container">
                                <h3 className="image-header">
                                    <span>Original {!isImageLoaded && '(Loading...)'}</span>
                                    <span className="header-controls">
                                        {zoomScale > 1 && (
                                            <span className="zoom-indicator">
                                                {Math.round(zoomScale * 100)}%
                                                <button className="reset-zoom-button" onClick={handleResetZoom}>
                                                    Reset
                                                </button>
                                            </span>
                                        )}
                                        <div className="zoom-controls">
                                            <button
                                                onClick={handleZoomIn}
                                                disabled={zoomScale >= 10}
                                                title="Zoom In"
                                            >
                                                +
                                            </button>
                                            <button
                                                onClick={handleZoomOut}
                                                disabled={zoomScale <= 1}
                                                title="Zoom Out"
                                            >
                                                -
                                            </button>
                                        </div>
                                    </span>
                                </h3>

                                <div
                                    className={`image-wrapper ${isDrawingRect ? 'rectangle-active' : ''} ${zoomScale > 1 ? 'zoomed' : ''}`}
                                    style={{
                                        position: 'relative',
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        cursor: zoomScale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
                                    }}
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    onMouseLeave={() => {
                                        if (isDragging) {
                                            setIsDragging(false);
                                        }
                                        if (isDrawingRect) {
                                            cancelRectangleMode();
                                        }
                                    }}
                                    onContextMenu={handleContextMenu}
                                >
                                    {/* Conteneur pour le zoom - TOUT doit être à l'intérieur */}
                                    <div
                                        className="zoom-container"
                                        style={{
                                            ...getZoomStyle(),
                                            width: '100%',
                                            height: '100%',
                                            display: 'flex',
                                            justifyContent: 'center',
                                            alignItems: 'center',
                                            position: 'relative' // ← AJOUTER cette ligne
                                        }}
                                    >
                                        <img
                                            ref={imageRef}
                                            src={URL.createObjectURL(currentImage)}
                                            alt="Original"
                                            className={`correction-image ${isDrawingRect ? 'rectangle-mode' : 'clickable-image'}`}
                                            onError={(e) => console.error('Erreur de chargement Original:', e)}
                                            onLoad={handleImageLoad}
                                            style={{
                                                opacity: isImageLoaded ? 1 : 0.7,
                                                maxWidth: '100%',
                                                maxHeight: '100%',
                                                width: 'auto',
                                                height: 'auto',
                                            }}
                                        />

                                        {/* Points - DÉPLACÉS dans le conteneur de zoom */}
                                        {points.map(point => {
                                            const displayCoords = getDisplayCoordinates(point.relX, point.relY);
                                            return (
                                                <div
                                                    key={point.id}
                                                    className={`point ${point.type}-point`}
                                                    style={{
                                                        position: 'absolute',
                                                        left: displayCoords.x - 4,
                                                        top: displayCoords.y - 4,
                                                        // SUPPRIMER la transformation individuelle
                                                    }}
                                                    title={`${point.type} point`}
                                                />
                                            );
                                        })}

                                        {currentRect && (
                                            <div
                                                className="rectangle-preview"
                                                style={{
                                                    position: 'absolute',
                                                    left: currentRect.x,
                                                    top: currentRect.y,
                                                    width: currentRect.width,
                                                    height: currentRect.height,
                                                    border: '2px dashed #dc3545',
                                                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                                                    pointerEvents: 'none',
                                                    // SUPPRIMER la transformation individuelle
                                                }}
                                            />
                                        )}

                                        {rectangles.map((rect, index) => {
                                            const displayCoords = getDisplayCoordinates(rect.relX, rect.relY);
                                            const displayDims = getDisplayDimensions(rect.relWidth, rect.relHeight);

                                            return (
                                                <div
                                                    key={index}
                                                    className="rectangle-final"
                                                    style={{
                                                        position: 'absolute',
                                                        left: displayCoords.x,
                                                        top: displayCoords.y,
                                                        width: displayDims.width,
                                                        height: displayDims.height,
                                                        border: '2px solid #dc3545',
                                                        backgroundColor: 'rgba(220, 53, 69, 0.2)',
                                                        pointerEvents: 'none',
                                                        // SUPPRIMER la transformation individuelle
                                                    }}
                                                />
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="segmentation-steps-section">
                            <h3>Steps ({segmentationSteps.length})</h3>
                            <div className="steps-container">
                                {segmentationSteps.map(step => (
                                    <div
                                        key={step.id}
                                        className={`step-item ${selectedStepId === step.id ? 'selected' : ''}`}
                                        onClick={() => handleStepClick(step.id)}
                                    >
                                        <div className="step-image-container">
                                            <img
                                                src={step.imageUrl}
                                                alt={`Step ${step.id}`}
                                                className="step-image"
                                                onError={(e) => console.error('Erreur de chargement Step:', e)}
                                            />
                                            <button
                                                className="step-delete-button"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    removeSegmentationStep(step.id);
                                                }}
                                                title="Delete step"
                                            >
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                                    <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                                                </svg>
                                            </button>
                                        </div>
                                    </div>
                                ))}
                                {segmentationSteps.length === 0 && (
                                    <div className="no-steps-message">
                                        No steps yet
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="correction-sidebar">
                        <div className="correction-start">
                            <h3>Processing method</h3>

                            <div className="point-selector">
                                <label>
                                    <input
                                        type="radio"
                                        name="startType"
                                        value="segmented"
                                        checked={startType === 'segmented'}
                                        onChange={() => setStartType('segmented')}
                                    />
                                    Use PlantSAM Output
                                </label>

                                <label>
                                    <input
                                        type="radio"
                                        name="startType"
                                        value="scratch"
                                        checked={startType === 'scratch'}
                                        onChange={() => setStartType('scratch')}
                                    />
                                    Start from scratch
                                </label>
                            </div>
                        </div>

                        <div className="correction-controls">
                            <h3>Segmentation tools</h3>

                            <div className="point-selector">
                                <label>
                                    <input
                                        type="radio"
                                        name="pointType"
                                        value="positive"
                                        checked={pointType === 'positive'}
                                        onChange={() => setPointType('positive')}
                                    />
                                    Positive point
                                </label>

                                <label>
                                    <input
                                        type="radio"
                                        name="pointType"
                                        value="negative"
                                        checked={pointType === 'negative'}
                                        onChange={() => setPointType('negative')}
                                    />
                                    Negative point
                                </label>
                            </div>

                            <div className="action-buttons">
                                <button
                                    className="control-button full-segmentation-button"
                                    onClick={applyFullSegmentation}
                                    disabled={isProcessingFull}
                                >
                                    {isProcessingFull ? 'Processing...' : 'Full Segmentation'}
                                </button>

                                <button
                                    className="control-button manual-noise-removal-button"
                                    onClick={handleManualNoiseRemovalClick}
                                    disabled={isDrawingRect || !isImageLoaded}
                                >
                                    {isDrawingRect ? 'Drawing Rectangle...' :
                                        !isImageLoaded ? 'Waiting for image...' : 'Manual Noise Removal'}
                                </button>

                                <button
                                    className="control-button undo-button"
                                    onClick={undoLastPoint}
                                    disabled={points.length === 0}
                                >
                                    Undo point
                                </button>

                                <button
                                    className="control-button clear-button"
                                    onClick={clearPoints}
                                    disabled={points.length === 0}
                                >
                                    Clear all points
                                </button>

                                <button
                                    className="control-button download-button"
                                    onClick={downloadProcessedImage}
                                    disabled={!processedImageUrl}
                                >
                                    Download final mask
                                </button>
                            </div>
                        </div>

                        <div className="correction-methods">
                            <h3>Processing Algorithm</h3>

                            <div className="point-selector">
                                <label>
                                    <input
                                        type="radio"
                                        name="algoType"
                                        value="union"
                                        checked={algoType === 'union'}
                                        onChange={() => setAlgoType('union')}
                                    />
                                    Union
                                </label>

                                <label>
                                    <input
                                        type="radio"
                                        name="algoType"
                                        value="intersection"
                                        checked={algoType === 'intersection'}
                                        onChange={() => setAlgoType('intersection')}
                                    />
                                    Intersection
                                </label>

                                <label>
                                    <input
                                        type="radio"
                                        name="algoType"
                                        value="iou"
                                        checked={algoType === 'iou'}
                                        onChange={() => setAlgoType('iou')}
                                    />
                                    IOU
                                </label>
                            </div>

                            <button
                                className="control-button apply-algorithm-button"
                                onClick={applyAlgorithm}
                                disabled={isLoading}
                            >
                                {isLoading ? `Applying ${algoType}...` : `Apply ${algoType}`}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default CorrectionPage