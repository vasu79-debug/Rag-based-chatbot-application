import React, { useState, useEffect, useCallback } from "react";
import Header from "./components/Header";
import ChatView from "./components/ChatView";
import AdminPortal from "./components/AdminPortal";
import CitationsDrawer from "./components/CitationsDrawer";
import {
  fetchHealth,
  fetchDocuments,
  streamChatMessage,
  uploadDocument,
  ingestUrl,
  deleteDocument,
} from "./api";

const getInitialTab = () => {
  if (typeof window === "undefined") return "chat";
  const path = window.location.pathname.toLowerCase();
  const hash = window.location.hash.toLowerCase();
  const search = window.location.search.toLowerCase();

  if (
    path.endsWith("/admin") ||
    hash === "#/admin" ||
    hash === "#admin" ||
    search.includes("admin=true") ||
    search.includes("tab=admin")
  ) {
    return "admin";
  }
  return "chat";
};

export default function App() {
  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);

  // Sync route on popstate and hashchange
  useEffect(() => {
    const handleLocationChange = () => {
      setActiveTab(getInitialTab());
    };

    window.addEventListener("popstate", handleLocationChange);
    window.addEventListener("hashchange", handleLocationChange);
    return () => {
      window.removeEventListener("popstate", handleLocationChange);
      window.removeEventListener("hashchange", handleLocationChange);
    };
  }, []);

  const handleNavigate = useCallback((tab) => {
    setActiveTab(tab);
    if (tab === "admin") {
      if (window.location.hash !== "#/admin") {
        window.history.pushState(null, "", "#/admin");
      }
    } else {
      const url = new URL(window.location.href);
      let changed = false;

      // Clean up path
      if (url.pathname.endsWith("/admin")) {
        url.pathname = url.pathname.replace(/\/admin$/, "") || "/";
        changed = true;
      }
      // Clean up hash
      if (url.hash.includes("admin")) {
        url.hash = "";
        changed = true;
      }
      // Clean up search params
      if (url.searchParams.has("admin") || url.searchParams.get("tab") === "admin") {
        url.searchParams.delete("admin");
        if (url.searchParams.get("tab") === "admin") {
          url.searchParams.delete("tab");
        }
        changed = true;
      }

      if (changed || window.location.hash) {
        window.history.pushState(null, "", url.pathname + url.search);
      }
    }
  }, []);

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setCurrentStage(null);
    setLoading(false);
  }, []);

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
    setCurrentStage({ stage: "routing", step: "1/3", label: "Routing · Analyzing query intent & scope..." });

    try {
      const historyPayload = messages.map((m) => {
        let textContent = m.content || "";
        if (m.role === "assistant" && m.payload) {
          const parts = [];
          if (m.payload.org_section?.content) {
            parts.push(`[Internal Knowledge]: ${m.payload.org_section.content}`);
          }
          if (m.payload.general_section?.content) {
            parts.push(`[General Knowledge]: ${m.payload.general_section.content}`);
          }
          textContent = parts.join("\n\n");
        }
        return {
          role: m.role,
          content: textContent,
        };
      });

      const res = await streamChatMessage(text, historyPayload, (stage) => {
        setCurrentStage(stage);
      });

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
            label: "⚠️ Response Notice",
            content: `Unable to retrieve response: ${err.message}`,
          },
        },
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setCurrentStage(null);
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

  const handleIngestUrl = async (url) => {
    setUploading(true);
    try {
      await ingestUrl(url);
      await loadData();
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm("Are you sure you want to delete this document from the knowledge base?")) {
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
        onNavigate={handleNavigate}
        onNewChat={handleNewChat}
        healthData={healthData}
      />

      <main className="main-content">
        {activeTab === "admin" ? (
          <AdminPortal
            documents={documents}
            onUpload={handleUpload}
            onIngestUrl={handleIngestUrl}
            onDelete={handleDelete}
            uploading={uploading}
          />
        ) : (
          <ChatView
            messages={messages}
            onSendMessage={handleSendMessage}
            loading={loading}
            currentStage={currentStage}
            onSelectCitation={setSelectedCitation}
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

