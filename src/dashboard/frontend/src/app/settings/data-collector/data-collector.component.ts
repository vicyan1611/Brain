import { CommonModule } from "@angular/common";
import { Component, OnDestroy, OnInit } from "@angular/core";
import { Subscription } from "rxjs";
import { WebSocketService } from "../../webSocket/web-socket.service";

interface DataCollectionPayload {
  state?: string;
  session?: string;
  frames?: number;
  zip_path?: string | null;
  message?: string;
  started_at?: number;
}

@Component({
  selector: "app-data-collector",
  standalone: true,
  imports: [CommonModule],
  templateUrl: "./data-collector.component.html",
  styleUrls: ["./data-collector.component.css"],
})
export class DataCollectorComponent implements OnInit, OnDestroy {
  isRecording = false;
  status = "Idle";
  frames = 0;
  session = "";
  message = "Ready to record";
  lastZipUrl = "";

  private statusSub?: Subscription;
  private ackSub?: Subscription;
  private backendUrl: string;

  constructor(private webSocketService: WebSocketService) {
    this.backendUrl = this.webSocketService.getBackendHttpUrl();
  }

  ngOnInit(): void {
    this.statusSub = this.webSocketService
      .receiveDataCollectionStatus()
      .subscribe((event: any) => {
        const payload: DataCollectionPayload = event?.value ?? event ?? {};
        this.status = payload.state ?? "Idle";
        this.isRecording = this.status === "recording";
        this.session = payload.session ?? this.session;
        this.frames = payload.frames ?? this.frames;
        this.message =
          payload.message ??
          (this.isRecording ? "Recording..." : "Ready to record");

        if (payload.zip_path) {
          this.lastZipUrl = `${this.backendUrl}/api/data-collection/latest`;
          this.message = "Data ready to download";
        }
      });

    this.ackSub = this.webSocketService
      .receiveDataCollectionAck()
      .subscribe((event: any) => {
        const payload = event?.data ?? event;
        if (payload?.Action === "start") {
          this.message = "Starting recording...";
        } else if (payload?.Action === "stop") {
          this.message = "Stopping recording...";
        }
      });
  }

  ngOnDestroy(): void {
    this.statusSub?.unsubscribe();
    this.ackSub?.unsubscribe();
  }

  startRecording(): void {
    if (this.isRecording) return;
    this.webSocketService.sendMessageToFlask(
      JSON.stringify({ Name: "DataCollection", Action: "start" }),
    );
    this.isRecording = true;
    this.status = "recording";
    this.message = "Starting recording...";
  }

  stopRecording(): void {
    if (!this.isRecording) return;
    this.webSocketService.sendMessageToFlask(
      JSON.stringify({ Name: "DataCollection", Action: "stop" }),
    );
    this.status = "idle";
    this.message = "Stopping recording...";
  }

  downloadLatest(): void {
    const url = `${this.backendUrl}/api/data-collection/latest`;
    window.open(url, "_blank");
  }
}
