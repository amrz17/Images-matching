import React, { useState } from 'react';

function TextInputEnter() {
  const [text, setText] = useState('');
  const [response, setResponse] = useState('');

  const handleChange = (e) => {
    setText(e.target.value);
  };

  const handleKeyDown = async (e) => {
    if (e.key === 'Enter') {
      try {
        const res = await fetch('http://localhost:5000/upload-exit', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: text }),
        });

        const data = await res.json();
        setResponse(data.message || 'Berhasil!');
        setText(''); // Hapus input setelah kirim
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
    </div>
  );
}

export default TextInputEnter;
