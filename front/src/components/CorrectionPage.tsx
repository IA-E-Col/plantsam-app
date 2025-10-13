import { useState, useRef, useEffect } from 'react'
import './CorrectionPage.css'

interface Point {
    x: number;
    y: number;
    type: 'positive' | 'negative';
    id: number;
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
    const [pointType, setPointType] = useState<'positive' | 'negative'>('positive')
    const [algoType, setAlgoType] = useState<'union' | 'intersection' | 'iou'>('union')
    const [startType, setStartType] = useState<'segmented' | 'scratch'>('segmented')
    const [points, setPoints] = useState<Point[]>([])
    const [processedImageUrl, setProcessedImageUrl] = useState<string>('')
    const [isLoading, setIsLoading] = useState(false)
    const [segmentationSteps, setSegmentationSteps] = useState<SegmentationStep[]>([])
    const [initialSegmentationUrl, setInitialSegmentationUrl] = useState<string>('')
    const [selectedStepId, setSelectedStepId] = useState<number | null>(null)
    const [isProcessingFull, setIsProcessingFull] = useState(false)
    const imageRef = useRef<HTMLImageElement>(null)
    const pointIdCounter = useRef(0)
    const stepIdCounter = useRef(0)

    const currentImage = images[currentImageIndex]

    const loadProcessedImage = async () => {
        if (!groupId) return

        setIsLoading(true)
        try {
            const response = await fetch(`/api/files/group/${groupId}/${currentImageIndex}/result`)

            if (response.ok) {
                const blob = await response.blob()
                if (processedImageUrl) {
                    URL.revokeObjectURL(processedImageUrl)
                }
                const url = URL.createObjectURL(blob)
                setProcessedImageUrl(url)
                console.log('Image traitée chargée:', url)
            } else {
                const url = URL.createObjectURL(currentImage)
                setProcessedImageUrl(url)
                console.log('Image originale utilisée comme fallback')
            }
        } catch (error) {
            console.error('Erreur lors du chargement de l\'image traitée:', error)
            const url = URL.createObjectURL(currentImage)
            setProcessedImageUrl(url)
        } finally {
            setIsLoading(false)
        }
    }

    const loadInitialSegmentation = async () => {
        if (!groupId) return

        try {
            const response = await fetch(`/api/files/group/${groupId}/${currentImageIndex}/result`)
            if (response.ok) {
                const blob = await response.blob()
                if (initialSegmentationUrl) {
                    URL.revokeObjectURL(initialSegmentationUrl)
                }
                const url = URL.createObjectURL(blob)
                setInitialSegmentationUrl(url)
                console.log('Segmentation initiale chargée:', url)
            }
        } catch (error) {
            console.error('Erreur lors du chargement de la segmentation initiale:', error)
        }
    }

    useEffect(() => {
        return () => {
            if (processedImageUrl) {
                URL.revokeObjectURL(processedImageUrl)
            }
            if (initialSegmentationUrl) {
                URL.revokeObjectURL(initialSegmentationUrl)
            }
            segmentationSteps.forEach(step => {
                URL.revokeObjectURL(step.imageUrl)
            })
        }
    }, [processedImageUrl, initialSegmentationUrl, segmentationSteps])

    useEffect(() => {
        if (groupId && images.length > 0) {
            loadProcessedImage()
            loadInitialSegmentation()
            setPoints([])
            setSelectedStepId(null)
            const initialUrl = URL.createObjectURL(currentImage)
            setSegmentationSteps([{
                id: stepIdCounter.current++,
                imageUrl: initialUrl,
                stepName: 'initial',
                timestamp: new Date()
            }])
            console.log('Initialisation avec image:', initialUrl)
        }
    }, [groupId, currentImageIndex, images.length])

    const removeSegmentationStep = (id: number) => {
        setSegmentationSteps(prev => {
            const stepToRemove = prev.find(step => step.id === id)
            if (stepToRemove) {
                URL.revokeObjectURL(stepToRemove.imageUrl)
            }
            return prev.filter(step => step.id !== id)
        })

        if (selectedStepId === id) {
            setSelectedStepId(null)
        }
    }

    const saveStepImage = async (imageBlob: Blob, stepName: string) => {
        if (!groupId) return false;

        try {
            const imageBuffer = await imageBlob.arrayBuffer();
            const response = await fetch(
                `/api/files/group/${groupId}/${currentImageIndex}/save_step?stepName=${encodeURIComponent(stepName)}`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/octet-stream',
                    },
                    body: imageBuffer
                }
            );

            if (response.ok) {
                console.log(`Étape "${stepName}" sauvegardée avec succès`);
                return true;
            } else {
                console.error('Erreur lors de la sauvegarde de l\'étape:', response.status);
                return false;
            }
        } catch (error) {
            console.error('Erreur lors de la sauvegarde de l\'étape:', error);
            return false;
        }
    };

    const applyFullSegmentation = async () => {
        if (!groupId) return;

        setIsProcessingFull(true);
        try {
            console.log('Application de la segmentation totale...');
            const response = await fetch(`/api/files/group/${groupId}/${currentImageIndex}/process`, {
                method: 'POST'
            });

            if (response.ok) {
                console.log('Segmentation totale réussie');

                // Recharger l'image traitée
                await loadProcessedImage();

                // Ajouter une étape pour la segmentation totale
                const stepName = `full_segmentation_${Date.now()}`;
                const fullSegResponse = await fetch(`/api/files/group/${groupId}/${currentImageIndex}/result`);
                if (fullSegResponse.ok) {
                    const fullSegBlob = await fullSegResponse.blob();
                    await saveStepImage(fullSegBlob, stepName);

                    const fullSegUrl = URL.createObjectURL(fullSegBlob);
                    const newStep: SegmentationStep = {
                        id: stepIdCounter.current++,
                        imageUrl: fullSegUrl,
                        stepName: stepName,
                        timestamp: new Date()
                    };
                    setSegmentationSteps(prev => [...prev, newStep]);
                }

                // Réinitialiser les points
                setPoints([]);

                // Pas d'alerte de confirmation comme demandé
            } else {
                const errorText = await response.text();
                console.error('Erreur lors de la segmentation totale:', response.status, errorText);
                alert(`Erreur lors de la segmentation totale: ${response.status}`);
            }
        } catch (error) {
            console.error('Erreur API lors de la segmentation totale:', error);
            alert(`Erreur lors de la segmentation totale: ${error.message}`);
        } finally {
            setIsProcessingFull(false);
        }
    };

    const applySegmentationWithPoints = async (pointsList: Point[]) => {
        if (!groupId) return;

        setIsLoading(true);
        try {
            let unionResponse: Response | null = null;
            let individualResponse: Response | null = null;
            let stepName = '';

            if (algoType === 'union') {
                const positivePoints = pointsList.filter(p => p.type === 'positive');
                const negativePoints = pointsList.filter(p => p.type === 'negative');

                if (positivePoints.length === 0) {
                    if (negativePoints.length > 0 || pointsList.length === 0) {
                        await clearPoints();
                    }
                    setIsLoading(false);
                    return;
                }

                const lastPositivePoint = positivePoints[positivePoints.length - 1];

                if (imageRef.current) {
                    const img = imageRef.current;
                    const rect = img.getBoundingClientRect();
                    const naturalWidth = img.naturalWidth;
                    const naturalHeight = img.naturalHeight;
                    const displayedWidth = rect.width;
                    const displayedHeight = rect.height;
                    const scaleX = naturalWidth / displayedWidth;
                    const scaleY = naturalHeight / displayedHeight;

                    const scaledPoint = {
                        x: Math.round(lastPositivePoint.x * scaleX),
                        y: Math.round(lastPositivePoint.y * scaleY)
                    };

                    const unionUrl = `/api/files/group/${groupId}/${currentImageIndex}/segment_union?x=${scaledPoint.x}&y=${scaledPoint.y}&pointCount=${positivePoints.length}&startType=${startType}`;
                    console.log('URL de segmentation union:', unionUrl);
                    unionResponse = await fetch(unionUrl, { method: 'POST' });

                    const individualUrl = `/api/files/group/${groupId}/${currentImageIndex}/segment_with_points`;
                    const formData = new FormData();
                    formData.append('positivePoints', JSON.stringify([[scaledPoint.x, scaledPoint.y]]));
                    formData.append('negativePoints', JSON.stringify([]));
                    formData.append('startType', startType);

                    individualResponse = await fetch(individualUrl, {
                        method: 'POST',
                        body: formData
                    });

                    stepName = `step_individual_${positivePoints.length}_${Date.now()}`;
                }
            }
            else if (algoType === 'intersection') {
                const positivePoints = pointsList.filter(p => p.type === 'positive');
                const negativePoints = pointsList.filter(p => p.type === 'negative');

                if (negativePoints.length === 0) {
                    if (pointsList.length === 0) {
                        await clearPoints();
                    }
                    setIsLoading(false);
                    return;
                }

                const lastNegativePoint = negativePoints[negativePoints.length - 1];

                if (imageRef.current) {
                    const img = imageRef.current;
                    const rect = img.getBoundingClientRect();
                    const naturalWidth = img.naturalWidth;
                    const naturalHeight = img.naturalHeight;
                    const displayedWidth = rect.width;
                    const displayedHeight = rect.height;
                    const scaleX = naturalWidth / displayedWidth;
                    const scaleY = naturalHeight / displayedHeight;

                    const scaledPoint = {
                        x: Math.round(lastNegativePoint.x * scaleX),
                        y: Math.round(lastNegativePoint.y * scaleY)
                    };

                    const intersectionUrl = `/api/files/group/${groupId}/${currentImageIndex}/segment_intersection?x=${scaledPoint.x}&y=${scaledPoint.y}&pointCount=${negativePoints.length}&startType=${startType}`;
                    console.log('URL de segmentation intersection:', intersectionUrl);
                    unionResponse = await fetch(intersectionUrl, { method: 'POST' });

                    const individualUrl = `/api/files/group/${groupId}/${currentImageIndex}/segment_with_points`;
                    const formData = new FormData();
                    formData.append('positivePoints', JSON.stringify([]));
                    formData.append('negativePoints', JSON.stringify([[scaledPoint.x, scaledPoint.y]]));
                    formData.append('startType', startType);

                    individualResponse = await fetch(individualUrl, {
                        method: 'POST',
                        body: formData
                    });

                    stepName = `step_individual_${negativePoints.length}_${Date.now()}`;
                }
            }
            else {
                const positivePoints = pointsList.filter(p => p.type === 'positive').map(p => [p.x, p.y]);
                const negativePoints = pointsList.filter(p => p.type === 'negative').map(p => [p.x, p.y]);

                if (imageRef.current) {
                    const img = imageRef.current;
                    const rect = img.getBoundingClientRect();
                    const naturalWidth = img.naturalWidth;
                    const naturalHeight = img.naturalHeight;
                    const displayedWidth = rect.width;
                    const displayedHeight = rect.height;
                    const scaleX = naturalWidth / displayedWidth;
                    const scaleY = naturalHeight / displayedHeight;

                    const scaledPositivePoints = positivePoints.map(p => [
                        Math.round(p[0] * scaleX),
                        Math.round(p[1] * scaleY)
                    ]);
                    const scaledNegativePoints = negativePoints.map(p => [
                        Math.round(p[0] * scaleX),
                        Math.round(p[1] * scaleY)
                    ]);

                    const formData = new FormData();
                    formData.append('positivePoints', JSON.stringify(scaledPositivePoints));
                    formData.append('negativePoints', JSON.stringify(scaledNegativePoints));
                    formData.append('startType', startType);

                    unionResponse = await fetch(
                        `/api/files/group/${groupId}/${currentImageIndex}/segment_with_points`,
                        {
                            method: 'POST',
                            body: formData
                        }
                    );

                    individualResponse = unionResponse;
                    stepName = `step_iou_${pointsList.length}_${Date.now()}`;
                }
            }

            if (unionResponse && unionResponse.ok) {
                const unionBlob = await unionResponse.blob();

                if (!unionBlob.type.startsWith('image/')) {
                    const errorText = await unionBlob.text();
                    throw new Error(`Le serveur a renvoyé une erreur: ${errorText.substring(0, 100)}`);
                }

                if (unionBlob.size === 0) {
                    throw new Error('Image vide reçue du serveur');
                }

                if (processedImageUrl) {
                    URL.revokeObjectURL(processedImageUrl);
                }

                const unionUrl = URL.createObjectURL(unionBlob);
                console.log('Nouvelle URL union créée:', unionUrl);

                setProcessedImageUrl(unionUrl);
            } else {
                console.error('Erreur lors de la segmentation union:', unionResponse?.status);
                if (unionResponse) {
                    const errorText = await unionResponse.text();
                    console.error('Détails de l\'erreur:', errorText);
                }
            }

            if (individualResponse && individualResponse.ok && pointsList.length > 0) {
                const individualBlob = await individualResponse.blob();

                if (!individualBlob.type.startsWith('image/')) {
                    console.warn('Réponse non-image');
                } else if (individualBlob.size === 0) {
                    console.warn('Image vide');
                } else {
                    console.log('Sauvegarde de:', stepName);

                    await saveStepImage(individualBlob, stepName);

                    const individualUrl = URL.createObjectURL(individualBlob);

                    const newStep: SegmentationStep = {
                        id: stepIdCounter.current++,
                        imageUrl: individualUrl,
                        stepName: stepName,
                        timestamp: new Date()
                    };
                    setSegmentationSteps(prev => [...prev, newStep]);
                    console.log('Nouvelle étape individuelle ajoutée:', newStep);
                }
            }

            if (pointsList.length === 0) {
                const initialUrl = URL.createObjectURL(currentImage);
                setSegmentationSteps([{
                    id: stepIdCounter.current++,
                    imageUrl: initialUrl,
                    stepName: 'initial',
                    timestamp: new Date()
                }]);
            }
        } catch (error) {
            console.error('Erreur API:', error);
            alert(`Erreur lors de la segmentation: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleImageClick = async (e: React.MouseEvent<HTMLImageElement>) => {
        if (!imageRef.current || !groupId) return

        const img = imageRef.current
        const rect = img.getBoundingClientRect()

        const clickX = e.nativeEvent.offsetX
        const clickY = e.nativeEvent.offsetY

        console.log(`Clic sur l'image: (${clickX}, ${clickY})`)
        console.log(`Dimensions de l'image affichée: ${rect.width}x${rect.height}`)
        console.log(`Dimensions naturelles de l'image: ${img.naturalWidth}x${img.naturalHeight}`)

        const naturalWidth = img.naturalWidth
        const naturalHeight = img.naturalHeight
        const displayedWidth = rect.width
        const displayedHeight = rect.height

        const scaleX = naturalWidth / displayedWidth
        const scaleY = naturalHeight / displayedHeight

        const imageX = Math.round(clickX * scaleX)
        const imageY = Math.round(clickY * scaleY)

        console.log(`Coordonnées calculées en pixels: (${imageX}, ${imageY})`)
        console.log(`Facteurs d'échelle: scaleX=${scaleX}, scaleY=${scaleY}`)

        const displayX = clickX
        const displayY = clickY

        const newPoint: Point = {
            x: displayX,
            y: displayY,
            type: pointType,
            id: pointIdCounter.current++
        }

        const newPoints = [...points, newPoint]
        setPoints(newPoints)

        await applySegmentationWithPoints(newPoints)
    }

    const handleStepClick = (stepId: number) => {
        setSelectedStepId(stepId);
    }

    const undoLastPoint = async () => {
        if (points.length === 0) return

        const newPoints = points.slice(0, -1)
        setPoints(newPoints)

        if (newPoints.length === 0) {
            await clearPoints()
        } else {
            await applySegmentationWithPoints(newPoints)
        }
    }

    const clearPoints = async () => {
        try {
            const response = await fetch(
                `/api/files/group/${groupId}/${currentImageIndex}/clear_points`,
                {
                    method: 'POST'
                }
            )

            if (response.ok) {
                console.log('Points effacés avec succès')
                setPoints([])
                if (startType === 'segmented') {
                    await loadInitialSegmentation()
                    setProcessedImageUrl(initialSegmentationUrl)
                } else {
                    const url = URL.createObjectURL(currentImage)
                    setProcessedImageUrl(url)
                }
            } else {
                console.error('Erreur lors de effacement des points:', response.status)
            }
        } catch (error) {
            console.error('Erreur API:', error)
        }
    }

    const downloadProcessedImage = async () => {
        if (!processedImageUrl) {
            return
        }

        try {
            const response = await fetch(processedImageUrl)
            const blob = await response.blob()

            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.style.display = 'none'
            a.href = url

            const originalName = images[currentImageIndex].name
            const nameWithoutExtension = originalName.replace(/\.[^/.]+$/, "")
            const extension = 'png'
            a.download = `${nameWithoutExtension}_segmented_${Date.now()}.${extension}`

            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (error) {
            console.error('Erreur lors du téléchargement:', error)
        }
    }

    const previousImageUrl = selectedStepId
        ? segmentationSteps.find(step => step.id === selectedStepId)?.imageUrl
        : segmentationSteps.length > 1
            ? segmentationSteps[segmentationSteps.length - 2]?.imageUrl
            : URL.createObjectURL(currentImage);

    const segmentedImageUrl = processedImageUrl ||
        (segmentationSteps.length > 0
            ? segmentationSteps[segmentationSteps.length - 1]?.imageUrl
            : URL.createObjectURL(currentImage));

    console.log('URLs calculées:', {
        previousImageUrl,
        segmentedImageUrl,
        processedImageUrl,
        stepsCount: segmentationSteps.length,
        selectedStepId
    });

    if (!groupId) {
        return (
            <div className="correction-page">
                <div style={{ padding: '2rem', textAlign: 'center' }}>
                    <h2>Erreur: GroupId non défini</h2>
                    <button onClick={onBack}>Retour à l'accueil</button>
                </div>
            </div>
        )
    }

    return (
        <div className="correction-page">
            <header className="correction-header">
                <h1>PlantSAM</h1>
                <div>
                    <span style={{ marginRight: '1rem', color: '#666' }}>
                        Group: {groupId.substring(0, 8)}...
                    </span>
                    <button className="back-button" onClick={onBack}>
                        Back
                    </button>
                </div>
            </header>

            <div className="correction-layout">
                <div className="images-sidebar">
                    <h3>Images ({images.length})</h3>
                    <div className="images-list">
                        {images.map((image, index) => (
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
                            </div>
                        ))}
                    </div>
                </div>

                <div className="correction-content">
                    <div className="main-content-area">
                        <div className="images-section">
                            <div className="image-container">
                                <h3>Previous mask</h3>
                                <div className="image-wrapper">
                                    <img
                                        src={previousImageUrl}
                                        alt="Previous"
                                        className="correction-image"
                                        onError={(e) => console.error('Erreur de chargement Previous:', e)}
                                    />
                                </div>
                            </div>

                            <div className="image-container">
                                <h3>Final Mask {isLoading && '(Loading)'}</h3>
                                <div className="image-wrapper">
                                    <img
                                        src={segmentedImageUrl}
                                        alt="Segmented"
                                        className="correction-image"
                                        onError={(e) => console.error('Erreur de chargement Segmented:', e)}
                                    />
                                </div>
                            </div>

                            <div className="image-container">
                                <h3>Original</h3>
                                <div className="image-wrapper" style={{ position: 'relative' }}>
                                    <img
                                        ref={imageRef}
                                        src={URL.createObjectURL(currentImage)}
                                        alt="Original"
                                        className="correction-image clickable-image"
                                        onClick={handleImageClick}
                                        onError={(e) => console.error('Erreur de chargement Original:', e)}
                                    />

                                    {points.map(point => (
                                        <div
                                            key={point.id}
                                            className={`point ${point.type}-point`}
                                            style={{
                                                position: 'absolute',
                                                left: point.x - 4,
                                                top: point.y - 4,
                                            }}
                                            title={`${point.type} point`}
                                        />
                                    ))}
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
                                    className="control-button rectangle-button"
                                    onClick={() => console.log('Rectangle mode activated')}
                                >
                                    Manual noise removal
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
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default CorrectionPage