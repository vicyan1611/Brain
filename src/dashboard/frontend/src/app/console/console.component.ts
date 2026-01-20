import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnInit,
  OnDestroy,
  ViewChild,
  ElementRef,
  SimpleChanges,
  OnChanges,
  SecurityContext,
} from "@angular/core";
import { CommonModule } from "@angular/common";
import { WebSocketService } from "../webSocket/web-socket.service";
import { Subscription } from "rxjs";
import { DomSanitizer, SafeHtml } from "@angular/platform-browser";

@Component({
  selector: "app-console",
  standalone: true,
  imports: [CommonModule],
  templateUrl: "./console.component.html",
  styleUrls: ["./console.component.css"],
})
export class ConsoleComponent implements OnInit, OnDestroy, OnChanges {
  @Input() open: boolean = false;
  @Output() close = new EventEmitter<void>();
  @ViewChild("scrollContainer") private scrollContainer!: ElementRef;

  logs: SafeHtml[] = [];
  brainMonitorLogs: SafeHtml[] = [];
  private logSubscription: Subscription | undefined;
  private brainMonitorLogsSubscription: Subscription | undefined;
  currentLogType: "console" | "brainMonitor" = "console";

  constructor(
    private webSocketService: WebSocketService,
    private sanitizer: DomSanitizer,
  ) {}

  ngOnInit(): void {
    this.logSubscription = this.webSocketService
      .receiveConsoleLog()
      .subscribe((response: any) => {
        if (response && response.data) {
          this.addLog(response.data);
        }
      });

    this.brainMonitorLogsSubscription = this.webSocketService
      .receiveBrainMonitorLog()
      .subscribe((response: any) => {
        if (response && response.data) {
          this.brainMonitorLogs = [];
          response.data.forEach((log: string) => {
            this.addBrainMonitorLog(log);
          });
        } else if (response && response.error) {
          this.addBrainMonitorLog(`Error: ${response.error}`);
        }
      });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes["open"]) {
      if (this.open) {
        // Scroll to bottom when opening and disable page scroll
        setTimeout(() => this.scrollToBottom(), 100);
        document.body.style.overflow = "hidden";
      } else {
        document.body.style.overflow = "";
      }
    }
  }

  ngOnDestroy(): void {
    // Re-enable scroll if component is destroyed
    document.body.style.overflow = "";
    if (this.logSubscription) {
      this.logSubscription.unsubscribe();
    }

    if (this.brainMonitorLogsSubscription) {
      this.brainMonitorLogsSubscription.unsubscribe();
    }
  }

  switchLogType(type: "console" | "brainMonitor"): void {
    this.currentLogType = type;
    if (type === "brainMonitor" && this.brainMonitorLogs.length === 0) {
      this.webSocketService.requestBrainMonitorLog();
    }
    setTimeout(() => this.scrollToBottom(), 100);
  }

  private addBrainMonitorLog(message: string): void {
    const formattedMessage = this.parseAnsi(message.trim());
    this.brainMonitorLogs.push(formattedMessage);
    if (this.brainMonitorLogs.length > 500) {
      this.brainMonitorLogs.shift();
    }
    setTimeout(() => this.scrollToBottom(), 50);
  }

  private addLog(message: string): void {
    const timestamp = new Date().toLocaleTimeString();

    // Check if user is near bottom before adding log
    const isNearBottom = this.isUserNearBottom();

    const formattedMessage = this.parseAnsi(`[${timestamp}] ${message}`);
    this.logs.push(formattedMessage);

    // Keep only last 500 logs to prevent memory issues
    if (this.logs.length > 500) {
      this.logs.shift();
    }

    if (isNearBottom) {
      setTimeout(() => this.scrollToBottom(), 50);
    }
  }

  private parseAnsi(text: string): SafeHtml {
    // Basic parser for ANSI color codes
    // \x1b[...m

    let html = text.replace(/\x1b\[([0-9;]*)m/g, (match, codes) => {
      const style: string[] = [];
      const codeArray = codes.split(";").map(Number);

      for (const code of codeArray) {
        if (code === 0) {
          return "</span>"; // Reset
        } else if (code === 1) {
          style.push("font-weight: bold");
        } else if (code >= 30 && code <= 37) {
          const colors = [
            "black",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "white",
          ];
          style.push(`color: ${colors[code - 30]}`);
        } else if (code >= 90 && code <= 97) {
          const colors = [
            "grey",
            "#ff5555",
            "#55ff55",
            "#ffff55",
            "#5555ff",
            "#ff55ff",
            "#55ffff",
            "white",
          ];
          style.push(`color: ${colors[code - 90]}`);
        }
      }

      if (style.length > 0) {
        return `<span style="${style.join(";")}">`;
      }
      return "";
    });

    // Ensure we close any open spans (simple heuristic, not perfect nesting but works for flat colors)
    const openSpans = (html.match(/<span/g) || []).length;
    const closeSpans = (html.match(/<\/span>/g) || []).length;
    html += "</span>".repeat(Math.max(0, openSpans - closeSpans));

    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  private isUserNearBottom(): boolean {
    if (!this.scrollContainer) return true;

    // Increase threshold to account for layout shifts or fast scrolling
    const threshold = 150;
    const position =
      this.scrollContainer.nativeElement.scrollTop +
      this.scrollContainer.nativeElement.offsetHeight;
    const height = this.scrollContainer.nativeElement.scrollHeight;

    return position > height - threshold;
  }

  handleClose(): void {
    this.close.emit();
  }

  scrollToBottom(): void {
    try {
      if (this.scrollContainer) {
        this.scrollContainer.nativeElement.scrollTop =
          this.scrollContainer.nativeElement.scrollHeight;
      }
    } catch (err) {}
  }

  clearConsole(): void {
    if (this.currentLogType === 'console') {
        this.logs = [];
    } else {
        this.brainMonitorLogs = [];
    }
  }
}
