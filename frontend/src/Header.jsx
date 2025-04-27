import React from 'react';

function Header() {
  return (
    <header style={styles.header}>
      <h1 style={styles.title}>VEHICLE INSPECTION</h1>
      <h1 style={styles.title}>FOR SAFETY</h1>
    </header>
  );
}

const styles = {
  header: {
    // backgroundColor: '#4CAF50',
    backgroundColor: '#888',
    padding: '20px 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  title: {
    color: '#fff',
    margin: 0
  },
  navList: {
    listStyle: 'none',
    display: 'flex',
    gap: '15px',
    margin: 0,
    padding: 0
  },
  navItem: {
    color: '#fff',
    textDecoration: 'none',
    fontWeight: 'bold'
  }
};

export default Header;
