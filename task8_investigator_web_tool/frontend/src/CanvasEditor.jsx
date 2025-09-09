import React, { useRef, useState, useEffect } from 'react';
import { Stage, Layer, Rect, Image as KonvaImage, Transformer } from 'react-konva';
import useImage from 'use-image';
import { applyBlackout, applyBlur, detectObjects } from './api';
import './CanvasEditor.css';
import { FaMousePointer, FaCrop, FaTimes, FaFillDrip, FaMagic, FaDownload, FaRobot, FaUndo, FaSearchPlus, FaSearchMinus, FaHandPaper } from 'react-icons/fa';

function CanvasEditor({ image }) {
  const [img] = useImage(image);
  const [rectangles, setRectangles] = useState([]);
  const [tool, setTool] = useState('select');
  const [newRect, setNewRect] = useState(null);
  const [editedImage, setEditedImage] = useState(null);
  const [previewImg, setPreviewImg] = useState(null);
  const [stageScale, setStageScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  const stageRef = useRef();
  const transformerRef = useRef();
  const [stageSize, setStageSize] = useState({ width: window.innerWidth * 0.8, height: window.innerHeight * 0.8 });

  useEffect(() => {
    const handleResize = () => {
      setStageSize({
        width: window.innerWidth * 0.8,
        height: window.innerHeight * 0.8,
      });
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (img) {
      const scaleX = stageSize.width / img.width;
      const scaleY = stageSize.height / img.height;
      const fitScale = Math.min(scaleX, scaleY);
      setStageScale(fitScale);
      setStagePos({
        x: (stageSize.width - img.width * fitScale) / 2,
        y: (stageSize.height - img.height * fitScale) / 2,
      });
      setEditedImage(image);
    }
  }, [img, image, stageSize]);

  useEffect(() => {
    if (!editedImage) return;
    const imgObj = new window.Image();
    imgObj.src = editedImage;
    imgObj.onload = () => setPreviewImg(imgObj);
  }, [editedImage]);

  useEffect(() => {
    if (transformerRef.current && selectedId !== null) {
      const node = stageRef.current.findOne(`#rect${selectedId}`);
      transformerRef.current.nodes(node ? [node] : []);
      transformerRef.current.getLayer().batchDraw();
    }
  }, [selectedId, rectangles]);

  const rectIntersect = (r1, r2) =>
    !(r2.x > r1.x + r1.width || r2.x + r2.width < r1.x || r2.y > r1.y + r1.height || r2.y + r2.height < r1.y);

  const handleMouseDown = e => {
   if (tool === 'select' || tool === 'deselect') {
      const layer = stageRef.current.findOne('Layer');
      const pointer = stageRef.current.getRelativePointerPosition(layer);
    
      setNewRect({
        x: pointer.x,
        y: pointer.y,
        width: 0,
        height: 0
      });
    } else if (tool === 'hand') {
      stageRef.current.draggable(true);
    }
  };

const handleMouseMove = e => {
  if (!newRect) return;
  const layer = stageRef.current.findOne('Layer');
   const pointer = stageRef.current.getRelativePointerPosition(layer);
 
 setNewRect({
    ...newRect,
    width: pointer.x - newRect.x,
    height: pointer.y - newRect.y
  });
};

const handleMouseUp = e => {
  const stage = e.target.getStage();
  if (tool === 'select' && newRect) {
    setRectangles([...rectangles, { ...newRect, tool, selected: true }]);
    setNewRect(null);
  } else if (tool === 'deselect' && newRect) {
    setRectangles(rectangles.filter(r => !rectIntersect(r, newRect)));
    setNewRect(null);
  } else if (tool === 'hand') {
    stage.draggable(false);
  }
};

  const toggleSelect = index => {
    setSelectedId(index);
    setRectangles(rectangles.map((r, i) => i === index ? { ...r, selected: !r.selected } : r));
  };

  const imageToFile = async src => {
    const blob = await (await fetch(src)).blob();
    return new File([blob], 'image.png', { type: blob.type });
  };

  const applyLocalEffect = effect => {
    if (!editedImage) return;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgObj = new window.Image();
    imgObj.src = editedImage;
    imgObj.onload = () => {
      canvas.width = imgObj.width;
      canvas.height = imgObj.height;
      ctx.drawImage(imgObj, 0, 0);
      rectangles.filter(r => r.selected).forEach(r => {
        if (effect === 'blackout') {
          ctx.fillStyle = 'black';
          ctx.fillRect(r.x, r.y, r.width, r.height);
        } else if (effect === 'blur') {
          ctx.filter = 'blur(7px)';
          ctx.drawImage(canvas, r.x, r.y, r.width, r.height, r.x, r.y, r.width, r.height);
          ctx.filter = 'none';
        }
      });
      const newDataUrl = canvas.toDataURL();
      setHistory([...history, editedImage]);
      setEditedImage(newDataUrl);
      const preview = new window.Image();
      preview.src = canvas.toDataURL();
      setPreviewImg(preview);
    };
  };

  const handleBlackout = () => applyLocalEffect('blackout');
  const handleBlur = () => applyLocalEffect('blur');

  const handleDetectObjects = async () => {
    const file = await imageToFile(editedImage);
    const detected = await detectObjects(file);
    setRectangles(detected.map(d => ({ ...d, tool: 'select', selected: false })));
  };

  const handleCrop = () => {
    const selectRects = rectangles.filter(r => r.selected);
    if (!selectRects.length) return;
    const x1 = Math.min(...selectRects.map(r => r.x));
    const y1 = Math.min(...selectRects.map(r => r.y));
    const x2 = Math.max(...selectRects.map(r => r.x + r.width));
    const y2 = Math.max(...selectRects.map(r => r.y + r.height));
    const canvas = document.createElement('canvas');
    canvas.width = x2 - x1;
    canvas.height = y2 - y1;
    const ctx = canvas.getContext('2d');
    const imgObj = new window.Image();
    imgObj.src = editedImage;
    imgObj.onload = () => {
      ctx.drawImage(imgObj, -x1, -y1);
      setHistory([...history, editedImage]);
      setEditedImage(canvas.toDataURL());
      setRectangles([]);
    };
  };

  const handleUndo = () => {
    if (!history.length) return;
    const prev = history[history.length - 1];
    setEditedImage(prev);
    setHistory(history.slice(0, -1));
  };

  const handleDownload = () => {
    if (!editedImage) return;
    const downloadImg = new window.Image();
    downloadImg.src = editedImage;
    downloadImg.crossOrigin = 'anonymous';
    downloadImg.onload = () => {
       const canvas = document.createElement('canvas');
       const MAX_WIDTH = 1200;
       const scale = Math.min(1, MAX_WIDTH / downloadImg.width);
       canvas.width = downloadImg.width * scale;
       canvas.height = downloadImg.height * scale;
       const ctx = canvas.getContext('2d');
       ctx.drawImage(downloadImg, 0, 0, canvas.width, canvas.height);
       canvas.toBlob((blob) => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'edited_image.jpg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
      }, 'image/jpeg', 0.8);
    };
  };

  const handleWheel = e => {
    e.evt.preventDefault();
    const scaleBy = 1.1;
    const stage = stageRef.current;
    const oldScale = stageScale;
    const mousePointTo = {
      x: (stage.getPointerPosition().x - stagePos.x) / oldScale,
      y: (stage.getPointerPosition().y - stagePos.y) / oldScale,
    };
    const newScale = e.evt.deltaY > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setStageScale(newScale);
    setStagePos({
      x: stage.getPointerPosition().x - mousePointTo.x * newScale,
      y: stage.getPointerPosition().y - mousePointTo.y * newScale,
    });
  };

  const zoomIn = () => setStageScale(stageScale * 1.2);
  const zoomOut = () => setStageScale(stageScale / 1.2);

  return (
    <div className="editor-wrapper">
      <div className="editor-container">
        <div className="sidebar">
          <div className="tool-buttons">
            <button className={tool==='select'?'active-tool':''} onClick={()=>setTool('select')} disabled={!!newRect}><FaMousePointer /> Select</button>
            <button className={tool==='deselect'?'active-tool':''} onClick={()=>setTool('deselect')} disabled={!!newRect}><FaTimes /> Deselect</button>
            <button className={tool==='hand'?'active-tool':''} onClick={()=>setTool('hand')}><FaHandPaper /> Hand</button>
            <button onClick={handleBlackout} disabled={!!newRect}><FaFillDrip /> Blackout</button>
            <button onClick={handleBlur} disabled={!!newRect}><FaMagic /> Blur</button>
            <button onClick={handleDetectObjects}><FaRobot /> AI Detect</button>
            <button onClick={handleCrop}><FaCrop /> Crop</button>
            <button onClick={handleUndo} disabled={!history.length}><FaUndo /> Undo</button>
            <button onClick={handleDownload}><FaDownload /> Download</button>
            <button onClick={zoomIn}><FaSearchPlus /> Zoom In</button>
            <button onClick={zoomOut}><FaSearchMinus /> Zoom Out</button>
          </div>
        </div>
        <div className="canvas-container">
          <Stage
            width={stageSize.width}
            height={stageSize.height}
            scaleX={stageScale}
            scaleY={stageScale}
            x={stagePos.x}
            y={stagePos.y}
            draggable={tool === 'hand'}
            onDragStart={() => { if (tool === 'hand') stageRef.current.container().style.cursor = 'grabbing'; }}
            onDragEnd={() => { if (tool === 'hand') stageRef.current.container().style.cursor = 'grab'; }}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            ref={stageRef}
            style={{ cursor: tool === 'hand' ? 'grab' : (tool === 'select' || tool === 'deselect') ? 'crosshair' : 'default' }}
          >
            <Layer>
              {previewImg && <KonvaImage image={previewImg} />}
              {rectangles.map((r, i) =>
                <Rect
                  key={i}
                  id={`rect${i}`}
                  x={r.x}
                  y={r.y}
                  width={r.width}
                  height={r.height}
                  stroke={r.selected?'green':'red'}
                  fill={r.selected ? 'rgba(0,255,0,0.2)' : 'rgba(255,0,0,0.1)'}
                  strokeWidth={3}
                  draggable
                  onClick={() => {
                    if (tool === 'select') toggleSelect(i);
                  }}
                  onTransformEnd={e => {
                    const node = e.target;
                    const newRects = rectangles.slice();
                    newRects[i] = {
                      ...newRects[i],
                      x: node.x(),
                      y: node.y(),
                      width: node.width() * node.scaleX(),
                      height: node.height() * node.scaleY(),
                    };
                    node.scaleX(1);
                    node.scaleY(1);
                    setRectangles(newRects);
                  }}
                  onDragEnd={e => {
                    const newRects = rectangles.slice();
                    newRects[i] = {
                      ...newRects[i],
                      x: e.target.x(),
                      y: e.target.y(),
                    };
                    setRectangles(newRects);
                  }}
                />
              )}
              <Transformer ref={transformerRef} rotateEnabled={false} />
              {newRect &&
                <Rect
                  x={newRect.x}
                  y={newRect.y}
                  width={newRect.width}
                  height={newRect.height}
                  stroke={tool==='select'?'green':'red'}
                  fill='rgba(0,0,255,0.1)'
                  strokeWidth={3}
                />
              }
            </Layer>
          </Stage>
        </div>
      </div>
    </div>
  );
}

export default CanvasEditor;
