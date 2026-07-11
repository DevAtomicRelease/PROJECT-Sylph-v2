import { useConfirmStore } from "../stores";
import "./ConfirmDialog.css";

export function ConfirmDialog() {
  const { showConfirm, actionName, details, respond } = useConfirmStore();

  if (!showConfirm) return null;

  // Format details for display
  const renderDetails = () => {
    if (actionName === "send_email") {
      return (
        <div className="confirm-details-list">
          <p><strong>To:</strong> {details.to || "(None)"}</p>
          <p><strong>Subject:</strong> {details.subject || "(No Subject)"}</p>
          <p className="confirm-body-preview"><strong>Body:</strong> {details.body || ""}</p>
        </div>
      );
    } else if (actionName === "create_event") {
      return (
        <div className="confirm-details-list">
          <p><strong>Title:</strong> {details.title || "(No Title)"}</p>
          <p><strong>Start:</strong> {details.start_iso || ""}</p>
          <p><strong>End:</strong> {details.end_iso || ""}</p>
          {details.description && <p><strong>Description:</strong> {details.description}</p>}
        </div>
      );
    } else if (actionName === "delete_event") {
      return (
        <div className="confirm-details-list">
          <p><strong>Event ID:</strong> {details.event_id || ""}</p>
        </div>
      );
    }
    return <pre className="confirm-raw-details">{JSON.stringify(details, null, 2)}</pre>;
  };

  const getActionLabel = () => {
    if (actionName === "send_email") return "Send Email";
    if (actionName === "create_event") return "Create Calendar Event";
    if (actionName === "delete_event") return "Delete Calendar Event";
    return actionName;
  };

  return (
    <div className="confirm-overlay" id="confirm-overlay">
      <div className="confirm-dialog" id="confirm-dialog">
        <h3 className="confirm-title">Sylph Action Required</h3>
        <p className="confirm-prompt">
          Sylph wants to perform: <strong>{getActionLabel()}</strong>. Do you allow this?
        </p>

        <div className="confirm-details">
          {renderDetails()}
        </div>

        <div className="confirm-actions">
          <button className="confirm-btn deny" onClick={() => respond(false)}>
            Deny
          </button>
          <button className="confirm-btn approve" onClick={() => respond(true)}>
            Allow
          </button>
        </div>
      </div>
    </div>
  );
}
