import './App.css';
import logo from './logo.png';
import React, { useState } from 'react';
import { IoSend } from "react-icons/io5";


function ChatBot() {
  const [messages, setMessages] = useState(['Hello I am your AI training assistant.', '']);
  const [inputValue, setInputValue] = useState('');

  const sendMessage = () => {
    // Example function to handle sending messages
    // You'll need to implement the logic for handling messages here
    const newMessage = inputValue;
    setMessages([...messages, newMessage]);
    setInputValue(''); // Clear input after sending
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      sendMessage(); // Call sendMessage when Enter is pressed
    }
  };

  return (
    <div className="card">
      <div id="header">
        <img src={logo} alt="Logo" />
      </div>
      <div id="message-section">
          {messages.map((message, index) => (
            <div key={index} className={`message ${index % 2 === 0? 'bot' : 'user'}`}>
              <span>{message}</span>
            </div>
          ))}
      </div>
      <div id="input-section">
        <input
          id="input"
          type="text"
          placeholder="Type a message"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          autoComplete="off"
          autoFocus
          onKeyPress={handleKeyPress} 
        />
        <button className="send" onClick={sendMessage}>
          <div className="circle">
            <IoSend size={20} color="#800080" />
          </div>
        </button>
      </div>
    </div>
  );
}

export default ChatBot;
