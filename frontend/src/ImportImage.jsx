import React, { useState } from 'react';

const styles = {
  photoBox: (imageName) => ({
    width: '300px',
    height: '300px',
    backgroundImage: `url(../${imageName})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    borderRadius: '10px',
    border: '2px solid #ccc',
    margin: '20px auto'
  })
};

function ImportImage() {
  const [currentImage, setCurrentImage] = useState('motor.jpg');

  return (
    <div>
      <h1>Gambar Dinamis</h1>
      <div style={styles.photoBox(currentImage)}></div>

      {/* Ganti gambar secara dinamis */}
      <button onClick={() => setCurrentImage('original_image.jpg')}>Mobil</button>
      <button onClick={() => setCurrentImage('bus.jpg')}>Bus</button>
    </div>
  );
}

export default ImportImage;
