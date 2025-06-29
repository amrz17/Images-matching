import { useState } from "react";

function TextInputDummy({ onSubmit }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Kirim nilai ke parent
    if (onSubmit) {
      onSubmit(value.toLowerCase()); // pastikan lowercase
    }

    setValue(""); // kosongkan input
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <input
        type="text"
        placeholder="Enter ticket code"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        style={styles.input}
      />
      <button type="submit" style={styles.button}>Submit</button>
    </form>
  );
}

const styles = {
  form: {
    marginTop: '20px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  input: {
    padding: '10px',
    fontSize: '16px',
    width: '250px',
    marginBottom: '10px',
    borderRadius: '8px',
    border: '1px solid #ccc',
  },
  button: {
    padding: '10px 20px',
    fontSize: '16px',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: '#4CAF50',
    color: 'white',
    cursor: 'pointer',
  },
};

export default TextInputDummy;
