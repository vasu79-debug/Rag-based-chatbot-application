import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import ChatView from "./components/ChatView";
import AdminPortal from "./components/AdminPortal";
import CitationsDrawer from "./components/CitationsDrawer";
import {
  fetchHealth,
  fetchDocuments,
  sendChatMessage,
  uploadDocument,
  deleteDocument,
} from "./api";

export default function App() {
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "admin"
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);

  // Load initial health & indexed documents
  const loadData = async () => {
    try {
      const [health, docs] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchDocuments().catch(() => []),
      ]);
      setHealthData(health);
      setDocuments(docs);
    } catch (e) {
      console.error("Initialization error:", e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSendMessage = async (text) => {
    if (!text.trim() || loading) return;

    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const historyPayload = messages.map((m) => ({
        role: m.role,
        content: m.content || "",
      }));

      const res = await sendChatMessage(text, historyPayload);

      const botMsg = {
        role: "assistant",
        payload: res,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        role: "assistant",
        payload: {
          general_section: {
            label: "⚠️ Processing Error",
            content: `Failed to generate response: ${err.message}`,
          },
        },
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (file) => {
    setUploading(true);
    try {
      await uploadDocument(file);
      await loadData();
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm("Are you sure you want to delete this document from the vector store?")) {
      return;
    }
    try {
      await deleteDocument(docId);
      await loadData();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthData={healthData}
      />

      <main className="main-content">
        {activeTab === "chat" ? (
          <ChatView
            messages={messages}
            onSendMessage={handleSendMessage}
            loading={loading}
            onSelectCitation={setSelectedCitation}
          />
        ) : (
          <AdminPortal
            documents={documents}
            onUpload={handleUpload}
            onDelete={handleDelete}
            uploading={uploading}
          />
        )}
      </main>

      <CitationsDrawer
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
