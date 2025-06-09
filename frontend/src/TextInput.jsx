import React, { useState } from 'react';

// const [response, setResponse] = useState('');
// const [imagePath, setImagePath] = useState('');

function TextInputEnter() {
  const [licensePlate, setLicensePlate] = useState('');
  const [matchStatus, setMatchStatus] = useState('');

  const [text, setText] = useState('');
  const [response, setResponse] = useState('');

  const handleChange = (e) => {
    setText(e.target.value);
  };

  const handleKeyDown = async (e) => {
    if (e.key === 'Enter') {
      try {
        const res = await fetch('http://localhost:5000/get-latest-qr', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: text }),
        });

        const data = await res.json();
         if (res.ok) {
          setResponse(data.message || 'Berhasil!');
          setLicensePlate(data.license_plate || '');
          // setImagePath(data.exit_image_path || '');
          setMatchStatus(data.match_status || '');
          setText(''); // Hapus input setelah kirim
        } else {
          setResponse(data.error || 'Terjadi kesalahan.');
        }
      } catch (error) {
        console.error('Error mengirim ke API:', error);
        setResponse('Gagal mengirim.');
      }
    }
  };

  return (
    <div className="p-4">
      <input
        type="text"
        value={text}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        style={{ border: 'none'}}
      />

      <p className="mt-4">{response}</p>
      {licensePlate && (
          <ExitResult
            licensePlate={licensePlate}
            matchStatus={matchStatus}
            // imagePath={imagePath}
            response={response}
          />
        )}
    </div>
  );
}

export default TextInputEnter;
