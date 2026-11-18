Fix silent send losses in the sidecar stdio path: byte-level stdin
reading with a max-line guard and a stall watchdog, send delivery
confirmation via send_ack/send_failed events, and the file_path send
caption is delivered instead of dropped (@j5bart)
