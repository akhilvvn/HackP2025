import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await axios.post(`${BASE_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error('Upload error:', error.response ? error.response.data : error);
    return null;
  }
};

export const applyBlackout = async (file, mask) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mask', JSON.stringify(mask));
  try {
    const response = await axios.post(`${BASE_URL}/blackout`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error('Blackout error:', error.response ? error.response.data : error);
    return null;
  }
};

export const applyBlur = async (file, mask) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('mask', JSON.stringify(mask));
  try {
    const response = await axios.post(`${BASE_URL}/blur`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error('Blur error:', error.response ? error.response.data : error);
    return null;
  }
};

export const detectObjects = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await axios.post(`${BASE_URL}/detect_objects`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error('DetectObjects error:', error.response ? error.response.data : error);
    return null;
  }
};
